import torch
import pandas as pd
from isaacgym.torch_utils import torch_rand_float, to_torch

from gym.utils.math import exp_avg_filter

from gym import LEGGED_GYM_ROOT_DIR
from gym.envs.mini_cheetah.mini_cheetah_osc import MiniCheetahOsc

MINI_CHEETAH_MASS = 8.292 * 9.81  # Weight of mini cheetah in Newtons


class MiniCheetahFloquet(MiniCheetahOsc):

    def _init_buffers(self):
        super()._init_buffers()

        self.last_phase_1 = torch.zeros(self.num_envs, 4, 2,
                                        device=self.device)
        self.last_phase_2 = torch.zeros(self.num_envs, 4, 2,
                                        device=self.device)
        self.last_phase_3 = torch.zeros(self.num_envs, 4, 2,
                                        device=self.device)
        self.last_phase_4 = torch.zeros(self.num_envs, 4, 2,
                                        device=self.device)
        self.crossings = torch.zeros_like(self.oscillators, dtype=torch.bool)
        self.floquet_reward = torch.zeros_like(self.crossings,
                                               dtype=torch.float32)

        self.average_frequency = (torch.ones_like(self.oscillators)
                                  * self.osc_omega)
        self.max_freq_diff = torch.zeros(self.num_envs, device=self.device)
        self.last_cross = torch.zeros_like(self.oscillators,
                                                   dtype=torch.long)

    def _pre_physics_step(self):
        super()._pre_physics_step()

    def _post_physics_step(self):
        super()._post_physics_step()
        self.max_freq_diff = self.average_frequency.max(dim=1)[0] \
                             - self.average_frequency.min(dim=1)[0]

    def _step_oscillators(self, dt=None):
        if dt is None:
            dt = self.dt

        local_feedback = self.osc_coupling * (torch.cos(self.oscillators)
                                              + self.osc_offset)
        grf = self._compute_grf()
        self.oscillators_vel = self.osc_omega - grf * local_feedback
        # self.oscillators_vel *= torch_rand_float(0.9,
        #                                          1.1,
        #                                          self.oscillators_vel.shape,
        #                                          self.device)
        self.oscillators_vel += (torch.randn(self.oscillators_vel.shape,
                                             device=self.device)
                                 * self.cfg.osc.process_noise_std)

        self.oscillators_vel *= 2*torch.pi
        self.oscillators += self.oscillators_vel*dt
        self.crossings = self.oscillators >= 2 * torch.pi
        self.oscillators = torch.remainder(self.oscillators, 2*torch.pi)

        for osc_id in range(4):
            env_id = self.crossings[:, osc_id].nonzero().flatten()
            if len(env_id) == 0:
                continue
            if osc_id == 0:
                self.last_phase_1.roll(1, dims=2)
                self.last_phase_1[env_id, :, 0] =\
                    self.oscillators[env_id, :]
            if osc_id == 1:
                self.last_phase_2.roll(1, dims=2)
                self.last_phase_2[env_id, :, 0] =\
                    self.oscillators[env_id, :]
            if osc_id == 2:
                self.last_phase_3.roll(1, dims=2)
                self.last_phase_3[env_id, :, 0] =\
                    self.oscillators[env_id, :]
            if osc_id == 3:
                self.last_phase_4.roll(1, dims=2)
                self.last_phase_4[env_id, :, 0] =\
                    self.oscillators[env_id, :]
            # * update average frequency of oscillators that reset
            self._update_avg_frequency(env_id, osc_id)

        self.oscillator_obs = torch.cat((torch.cos(self.oscillators),
                                         torch.sin(self.oscillators)), dim=1)

    def _update_avg_frequency(self, env_id, osc_id):
        time_steps = (self.common_step_counter
                      - self.last_cross[env_id, osc_id])
        self.average_frequency[env_id, osc_id] =\
            exp_avg_filter(1.0/(time_steps*self.dt),
                            self.average_frequency[env_id, osc_id])
        self.last_cross[env_id, osc_id] = self.common_step_counter

    def _reward_floquet(self):
        phase_diff = \
            torch.cat(((self.last_phase_1[:, :, 0] - self.last_phase_1[:, :, 1]).norm(dim=1).unsqueeze(1),
                       (self.last_phase_2[:, :, 0] - self.last_phase_2[:, :, 1]).norm(dim=1).unsqueeze(1),
                       (self.last_phase_3[:, :, 0] - self.last_phase_3[:, :, 1]).norm(dim=1).unsqueeze(1),
                       (self.last_phase_4[:, :, 0] - self.last_phase_4[:, :, 1]).norm(dim=1).unsqueeze(1)), dim=1)
        phase_diff /= 2*torch.pi
        self.floquet_reward = torch.where(self.crossings,
                                          phase_diff, self.floquet_reward)
        vel_error = (self.commands[:, :2] - self.base_lin_vel[:, :2]).norm(dim=1)
        vel_error = self._sqrdexp(vel_error/(self.scales["base_lin_vel"]/2.))
        # * when tracking is decent, penalize large deviations of the floquet multipliers
        reward = -self.floquet_reward.sum(dim=1)*vel_error
        return reward

    def _reward_locked_frequency(self):
        return -self.max_freq_diff


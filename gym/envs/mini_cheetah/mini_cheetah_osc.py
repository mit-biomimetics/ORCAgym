import torch
import pandas as pd
import numpy as np
from isaacgym.torch_utils import torch_rand_float, to_torch

from gym import LEGGED_GYM_ROOT_DIR
from gym.envs.mini_cheetah.mini_cheetah import MiniCheetah

from isaacgym import gymtorch, gymapi

MINI_CHEETAH_MASS = 8.292 * 9.81  # Weight of mini cheetah in Newtons

class MiniCheetahOsc(MiniCheetah):
    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        super().__init__(gym, sim, cfg, sim_params, sim_device, headless)
        self.process_noise_std = self.cfg.osc.process_noise_std

    def _init_buffers(self):
        super()._init_buffers()
        self.oscillators = torch.zeros(self.num_envs, 4,
                                       device=self.device)
        self.oscillator_obs = torch.zeros(self.num_envs, 8,
                                          device=self.device)

        self.oscillators_vel = torch.zeros_like(self.oscillators)
        self.grf = torch.zeros(self.num_envs, 4, device=self.device)
        self.osc_omega = self.cfg.osc.omega \
            * torch.ones(self.num_envs, 1, device=self.device)
        self.osc_coupling = self.cfg.osc.coupling \
            * torch.ones(self.num_envs, 1, device=self.device)
        self.osc_offset = self.cfg.osc.offset \
            * torch.ones(self.num_envs, 1, device=self.device)

    def _reset_oscillators(self, env_ids):
        if len(env_ids) == 0:
            return
                # * random
        if self.cfg.osc.init_to == 'random':
            self.oscillators[env_ids] = torch_rand_float(
                0, 2*torch.pi, shape=self.oscillators[env_ids].shape,
                device=self.device)
        elif self.cfg.osc.init_to == 'standing':
            self.oscillators[env_ids] = 3*torch.pi/2
        elif self.cfg.osc.init_to == 'trot':
            self.oscillators[env_ids] = torch.tensor([0., torch.pi, torch.pi, 0.],
                                                     device=self.device)
        elif self.cfg.osc.init_to == 'pace':
            self.oscillators[env_ids] = torch.tensor([0., torch.pi, 0., torch.pi],
                                                     device=self.device)
            if self.cfg.osc.init_w_offset:
                self.oscillators[env_ids, :] += \
                    torch.rand_like(self.oscillators[env_ids, 0]).unsqueeze(1) * 2 * torch.pi
        elif self.cfg.osc.init_to == 'pronk':
            self.oscillators[env_ids, :] *= 0.
        elif self.cfg.osc.init_to == 'bound':
            self.oscillators[env_ids, :] = torch.tensor([torch.pi, torch.pi, 0., 0.], device=self.device)
        else:
            raise NotImplementedError

        if self.cfg.osc.init_w_offset:
                self.oscillators[env_ids, :] += \
                    torch.rand_like(self.oscillators[env_ids, 0]).unsqueeze(1) * 2 * torch.pi
        self.oscillators = torch.remainder(self.oscillators, 2*torch.pi)


    def _reset_system(self, env_ids):
        if len(env_ids) == 0:
            return
        self._reset_oscillators(env_ids)

        self.oscillator_obs = torch.cat((torch.cos(self.oscillators),
                                         torch.sin(self.oscillators)), dim=1)

        # * keep some robots in the same starting state as they ended
        timed_out_subset = (self.timed_out & ~self.terminated) * \
            (torch.rand(self.num_envs, device=self.device)
             < self.cfg.init_state.timeout_reset_ratio)

        env_ids = (self.terminated | timed_out_subset).nonzero().flatten()
        if len(env_ids) == 0:
            return
        super()._reset_system(env_ids)

    def _pre_physics_step(self):
        super()._pre_physics_step()
        # self.grf = self._compute_grf()
        if not self.cfg.osc.randomize_osc_params:
            self.compute_osc_slope()

    def compute_osc_slope(self):
        cmd_x = torch.abs(self.commands[:, 0:1]) - self.cfg.osc.stop_threshold
        stop = (cmd_x < 0)

        self.osc_offset = stop * self.cfg.osc.offset
        self.osc_omega = stop * self.cfg.osc.omega_stop \
            + torch.randn_like(self.osc_omega) * self.cfg.osc.omega_var
        self.osc_coupling = stop * self.cfg.osc.coupling_stop \
            + torch.randn_like(self.osc_coupling) * self.cfg.osc.coupling_var

        self.osc_omega += (~stop) * torch.clamp(cmd_x*self.cfg.osc.omega_slope
                                                + self.cfg.osc.omega_step,
                                                min=0.,
                                                max=self.cfg.osc.omega_max)
        self.osc_coupling += \
            (~stop) * torch.clamp(cmd_x*self.cfg.osc.coupling_slope
                                  + self.cfg.osc.coupling_step,
                                  min=0.,
                                  max=self.cfg.osc.coupling_max)

        self.osc_omega = torch.clamp_min(self.osc_omega, 0.1)
        self.osc_coupling = torch.clamp_min(self.osc_coupling, 0)

    def _process_rigid_body_props(self, props, env_id):
        if env_id == 0:
            # * init buffers for the domain rand changes
            self.mass = torch.zeros(self.num_envs, 1, device=self.device)
            self.com = torch.zeros(self.num_envs, 3, device=self.device)

        # * randomize mass
        if self.cfg.domain_rand.randomize_base_mass:
            lower = self.cfg.domain_rand.lower_mass_offset
            upper = self.cfg.domain_rand.upper_mass_offset
            # self.mass_
            props[0].mass += np.random.uniform(lower, upper)
            self.mass[env_id] = props[0].mass
            # * randomize com position
            lower = self.cfg.domain_rand.lower_z_offset
            upper = self.cfg.domain_rand.upper_z_offset
            props[0].com.z += np.random.uniform(lower, upper)
            self.com[env_id, 2] = props[0].com.z

            lower = self.cfg.domain_rand.lower_x_offset
            upper = self.cfg.domain_rand.upper_x_offset
            props[0].com.x += np.random.uniform(lower, upper)
            self.com[env_id, 0] = props[0].com.x
        return props


    def _post_physics_step(self):
        """ Update all states that are not handled in PhysX """
        super()._post_physics_step()
        self.grf = self._compute_grf()
        # self._step_oscillators()

    def _post_torque_step(self):
        super()._post_torque_step()
        self._step_oscillators(self.dt/self.cfg.control.decimation)
        return None

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
        self.oscillators += self.oscillators_vel * dt  # torch.clamp(self.oscillators_vel * dt, min=0)
        self.oscillators = torch.remainder(self.oscillators, 2*torch.pi)
        self.oscillator_obs = torch.cat((torch.cos(self.oscillators),
                                         torch.sin(self.oscillators)), dim=1)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if len(env_ids) == 0:
            return
        super()._resample_commands(env_ids)
        possible_commands = torch.tensor(self.command_ranges["lin_vel_x"],
                                         device=self.device)
        self.commands[env_ids, 0:1] = possible_commands[torch.randint(
            0, len(possible_commands), (len(env_ids), 1),
            device=self.device)]
        # add some gaussian noise to the commands
        self.commands[env_ids, 0:1] += torch.randn((len(env_ids), 1),
                                                   device=self.device) \
                                        * self.cfg.commands.var

        # possible_commands = torch.tensor(self.command_ranges["lin_vel_y"],
        #                                  device=self.device)
        # self.commands[env_ids, 1:2] = possible_commands[torch.randint(
        #     0, len(possible_commands), (len(env_ids), 1),
        #     device=self.device)]
        # possible_commands = torch.tensor(self.command_ranges["yaw_vel"],
        #                                  device=self.device)
        # self.commands[env_ids, 0:1] = possible_commands[torch.randint(
        #     0, len(possible_commands), (len(env_ids), 1),
        #     device=self.device)]

        if (0 in self.cfg.commands.ranges.lin_vel_x):
            # * with 20% chance, reset to 0 commands except for forward
            self.commands[env_ids, 1:] *= (torch_rand_float(0, 1, (len(env_ids), 1),
                device=self.device).squeeze(1) < 0.8).unsqueeze(1)
            # * with 20% chance, reset to 0 commands except for rotation
            self.commands[env_ids, :2] *= (torch_rand_float(0, 1, (len(env_ids), 1),
                device=self.device).squeeze(1) < 0.8).unsqueeze(1)
            # * with 10% chance, reset to 0
            self.commands[env_ids, :] *= (torch_rand_float(0, 1, (len(env_ids), 1),
                device=self.device).squeeze(1) < 0.9).unsqueeze(1)

        if self.cfg.osc.randomize_osc_params:
            self._resample_osc_params(env_ids)

    def _resample_osc_params(self, env_ids):
        if (len(env_ids) > 0):
            self.osc_omega[env_ids, 0] = torch_rand_float(self.cfg.osc.omega_range[0],
                                                        self.cfg.osc.omega_range[1],
                                                        (len(env_ids), 1),
                                                        device=self.device).squeeze(1)
            self.osc_coupling[env_ids, 0] = torch_rand_float(self.cfg.osc.coupling_range[0],
                                                            self.cfg.osc.coupling_range[1],
                                                            (len(env_ids), 1),
                                                            device=self.device).squeeze(1)
            self.osc_offset[env_ids, 0] = torch_rand_float(self.cfg.osc.offset_range[0],
                                                        self.cfg.osc.offset_range[1],
                                                        (len(env_ids), 1),
                                                        device=self.device).squeeze(1)

    def perturb_base_velocity(self, velocity_delta, env_ids=None):
        if env_ids is None:
            env_ids = [range(self.num_envs)]
        self.root_states[env_ids, 7:10] += velocity_delta
        self.gym.set_actor_root_state_tensor(self.sim,
                                    gymtorch.unwrap_tensor(self.root_states))


    def _compute_grf(self, grf_norm=True):
        grf = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        if grf_norm:
            return torch.clamp_max(grf / MINI_CHEETAH_MASS, 1.0)
        else:
            return grf

    def _switch(self):
        c_vel = torch.linalg.norm(self.commands, dim=1)
        return torch.exp(-torch.square(torch.max(torch.zeros_like(c_vel),
                                                 c_vel-0.1))
                         / self.cfg.reward_settings.switch_scale)

    def _reward_cursorial(self):
        # penalize the abad joints being away from 0
        return -torch.mean(torch.square(self.dof_pos[:, 0:12:3]
                                        /self.scales["dof_pos"][0]), dim=1)

    def _reward_swing_grf(self):
        # Reward non-zero grf during swing (0 to pi)
        rew = self.get_swing_grf(self.cfg.osc.osc_bool, self.cfg.osc.grf_bool)
        return -torch.sum(rew, dim=1)

    def _reward_stance_grf(self):
        # Reward non-zero grf during stance (pi to 2pi)
        rew = self.get_stance_grf(self.cfg.osc.osc_bool, self.cfg.osc.grf_bool)
        return torch.sum(rew, dim=1)

    def get_swing_grf(self, osc_bool=False, contact_bool=False):
        if osc_bool:
            phase = torch.lt(self.oscillators, torch.pi).int()
        else:
            phase = torch.maximum(torch.zeros_like(self.oscillators),
                               torch.sin(self.oscillators))
        if contact_bool:
            return phase*torch.gt(self._compute_grf(),
                                  self.cfg.osc.grf_threshold)
        else:
            return phase*self._compute_grf()

    def get_stance_grf(self, osc_bool=False, contact_bool=False):
        if osc_bool:
            phase = torch.gt(self.oscillators, torch.pi).int()
        else:
            phase = torch.maximum(torch.zeros_like(self.oscillators),
                               - torch.sin(self.oscillators))
        if contact_bool:
            return phase*torch.gt(self._compute_grf(),
                                  self.cfg.osc.grf_threshold)
        else:
            return phase*self._compute_grf()

    def _reward_coupled_grf(self):
        """
        Multiply rewards for stance/swing grf, discount when undesirable
        behavior (grf during swing, no grf during stance)
        """
        swing_rew = self.get_swing_grf()
        stance_rew = self.get_stance_grf()
        combined_rew = self._sqrdexp(swing_rew*2) + stance_rew
        prod = torch.prod(torch.clip(combined_rew, 0, 1), dim=1)
        return prod - torch.ones_like(prod)

    def _reward_swing_velocity(self):
        """ Reward non-zero end effector velocity during swing (0 to pi) """
        # velocity = torch.tanh(torch.norm(self.end_effector_lin_vel, dim=-1))
        velocity = torch.zeros_like(self.oscillators)  # TODO: Grab velocity from AJ V2 code
        phase_bool = torch.lt(self.oscillators, torch.pi).int()
        phase_sin = torch.maximum(torch.zeros_like(self.oscillators),
                               torch.sin(self.oscillators))
        if self.cfg.osc.osc_bool:
            rew = phase_bool*velocity
        else:
            rew = phase_sin*velocity
        return torch.mean(rew, dim=1)

    def _reward_stance_velocity(self):
        """ Reward zero end effector velocity during swing (pi to 2pi) """
        # velocity = torch.tanh(torch.norm(self.end_effector_lin_vel, dim=-1))
        velocity = torch.zeros_like(self.oscillators)  # TODO: Grab velocity from AJ V2 code
        ph_bool = torch.gt(self.oscillators, torch.pi).int()
        ph_sin = torch.maximum(torch.zeros_like(self.oscillators),
                               - torch.sin(self.oscillators))
        if self.cfg.osc.osc_bool:
            rew = ph_bool*velocity
        else:
            rew = ph_sin*velocity
        # return torch.mean(self._sqrdexp(rew), dim=1)  # TODO: Higher reward if rew close to zero (how to decouple so that ignore if not in stance??)
        return - torch.mean(rew, dim=1)

    def _reward_dof_vel(self):
        return super()._reward_dof_vel()*self._switch()

    def _reward_dof_near_home(self):
        return super()._reward_dof_near_home()*self._switch()

    def _reward_stand_still(self):
        """Penalize motion at zero commands"""
        # * normalize angles so we care about being within 5 deg
        rew_pos = torch.mean(self._sqrdexp(
            (self.dof_pos - self.default_dof_pos)/torch.pi*36), dim=1)
        rew_vel = torch.mean(self._sqrdexp(self.dof_vel), dim=1)
        rew_base_vel = torch.mean(torch.square(self.base_lin_vel), dim=1)
        rew_base_vel += torch.mean(torch.square(self.base_ang_vel), dim=1)
        return (rew_vel+rew_pos-rew_base_vel)*self._switch()

    def _reward_standing_torques(self):
        """Penalize torques at zero commands"""
        return super()._reward_torques()*self._switch()

    # * gait similarity scores
    def angle_difference(self, theta1, theta2):
        diff = torch.abs(theta1 - theta2) % (2 * torch.pi)
        return torch.min(diff, 2*torch.pi - diff)

    def _reward_trot(self):
        # ! diagonal difference, front right and hind left
        angle = self.angle_difference(self.oscillators[:, 0],
                                      self.oscillators[:, 3])
        similarity = self._sqrdexp(angle, torch.pi)
        # ! diagonal difference, front left and hind right
        angle = self.angle_difference(self.oscillators[:, 1],
                                      self.oscillators[:, 2])
        similarity *= self._sqrdexp(angle, torch.pi)
        # ! diagonal difference, front left and hind right
        angle = self.angle_difference(self.oscillators[:, 0],
                                      self.oscillators[:, 1])
        similarity *= self._sqrdexp(angle - torch.pi, torch.pi)
        # ! diagonal difference, front left and hind right
        angle = self.angle_difference(self.oscillators[:, 2],
                                      self.oscillators[:, 3])
        similarity *= self._sqrdexp(angle - torch.pi, torch.pi)
        return similarity

    def _reward_pronk(self):
        # ! diagonal difference, front right and hind left
        angle = self.angle_difference(self.oscillators[:, 0],
                                      self.oscillators[:, 3])
        similarity = self._sqrdexp(angle, torch.pi)
        # ! diagonal difference, front left and hind right
        angle = self.angle_difference(self.oscillators[:, 1],
                                      self.oscillators[:, 2])
        similarity *= self._sqrdexp(angle, torch.pi)
        # ! difference, front right and front left
        angle = self.angle_difference(self.oscillators[:, 0],
                                      self.oscillators[:, 1])
        similarity *= self._sqrdexp(angle, torch.pi)
        # ! difference front right and hind right
        angle = self.angle_difference(self.oscillators[:, 0],
                                      self.oscillators[:, 2])
        similarity *= self._sqrdexp(angle, torch.pi)
        return similarity

    def _reward_pace(self):
        angle = self.angle_difference(self.oscillators[:, 0],
                                      self.oscillators[:, 2])
        similarity = self._sqrdexp(angle, torch.pi)
        # ! difference front left and hind left
        angle = self.angle_difference(self.oscillators[:, 1],
                                      self.oscillators[:, 3])
        similarity *= self._sqrdexp(angle, torch.pi)
        # ! difference front left and hind left
        angle = self.angle_difference(self.oscillators[:, 0],
                                      self.oscillators[:, 1])
        similarity *= self._sqrdexp(angle - torch.pi, torch.pi)
        # ! difference front left and hind left
        angle = self.angle_difference(self.oscillators[:, 2],
                                      self.oscillators[:, 3])
        similarity *= self._sqrdexp(angle - torch.pi, torch.pi)

        return similarity

    def _reward_any_symm_gait(self):
        rew_trot = self._reward_trot()
        rew_pace = self._reward_pace()
        rew_bound = self._reward_bound()
        return torch.max(torch.max(rew_trot, rew_pace), rew_bound)

    def _reward_enc_pace(self):
        return self._reward_pace()

    def _reward_bound(self):
        # ! difference, front right and front left
        angle = self.angle_difference(self.oscillators[:, 0],
                                      self.oscillators[:, 1])
        similarity = self._sqrdexp(angle, torch.pi)
        # ! difference hind right and hind left
        angle = self.angle_difference(self.oscillators[:, 2],
                                      self.oscillators[:, 3])
        similarity *= self._sqrdexp(angle, torch.pi)
        # ! difference right side
        angle = self.angle_difference(self.oscillators[:, 0],
                                      self.oscillators[:, 2])
        similarity *= self._sqrdexp(angle - torch.pi, torch.pi)
        # ! difference right side
        angle = self.angle_difference(self.oscillators[:, 1],
                                      self.oscillators[:, 3])
        similarity *= self._sqrdexp(angle - torch.pi, torch.pi)
        return similarity

    def _reward_asymettric(self):
        # ! hind legs
        angle = self.angle_difference(self.oscillators[:, 2],
                                      self.oscillators[:, 3])
        similarity = (1 - self._sqrdexp(angle, torch.pi))
        similarity *= (1 - self._sqrdexp((torch.pi - angle), torch.pi))
        # ! difference, left legs
        angle = self.angle_difference(self.oscillators[:, 1],
                                      self.oscillators[:, 3])
        similarity *= (1 - self._sqrdexp(angle, torch.pi))
        similarity *= (1 - self._sqrdexp((torch.pi - angle), torch.pi))
        # ! difference right legs
        angle = self.angle_difference(self.oscillators[:, 0],
                                      self.oscillators[:, 2])
        similarity *= (1 - self._sqrdexp(angle, torch.pi))
        similarity *= (1 - self._sqrdexp((torch.pi - angle), torch.pi))
        # ! front legs
        angle = self.angle_difference(self.oscillators[:, 0],
                                      self.oscillators[:, 1])
        similarity *= (1 - self._sqrdexp(angle, torch.pi))
        similarity *= (1 - self._sqrdexp((torch.pi - angle), torch.pi))
        # ! diagonal FR
        angle = self.angle_difference(self.oscillators[:, 0],
                                        self.oscillators[:, 3])
        similarity *= (1 - self._sqrdexp(angle, torch.pi))
        similarity *= (1 - self._sqrdexp((torch.pi - angle), torch.pi))
        # ! diagonal FL
        angle = self.angle_difference(self.oscillators[:, 1],
                                        self.oscillators[:, 2])
        similarity *= (1 - self._sqrdexp(angle, torch.pi))
        similarity *= (1 - self._sqrdexp((torch.pi - angle), torch.pi))
        return similarity
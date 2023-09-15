import torch
import pandas as pd
from isaacgym.torch_utils import torch_rand_float, to_torch

from gym import LEGGED_GYM_ROOT_DIR
from gym.envs.mini_cheetah.mini_cheetah import MiniCheetah


class MiniCheetahRef(MiniCheetah):
    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        csv_path = cfg.init_state.ref_traj.format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self.leg_ref = to_torch(pd.read_csv(csv_path).to_numpy(),
                                device=sim_device)
        self.omega = 2*torch.pi*cfg.control.gait_freq
        super().__init__(gym, sim, cfg, sim_params, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()
        self.phase = torch.zeros(self.num_envs, 1, dtype=torch.float,
                                 device=self.device)
        self.phase_obs = torch.zeros(self.num_envs, 2, dtype=torch.float,
                                     device=self.device)

    def _reset_system(self, env_ids):
        super()._reset_system(env_ids)
        self.phase[env_ids] = torch_rand_float(0, torch.pi,
                                               shape=self.phase[env_ids].shape,
                                               device=self.device)

    def _post_physics_step(self):
        """ Update all states that are not handled in PhysX """
        super()._post_physics_step()
        self.phase = torch.fmod(self.phase+self.dt*self.omega, 2*torch.pi)
        self.phase_obs = torch.cat((torch.sin(self.phase),
                                    torch.cos(self.phase)), dim=1)

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        # * with 10% chance, reset to 0 commands
        rand_ids = torch_rand_float(0, 1, (len(env_ids), 1),
                                    device=self.device).squeeze(1)
        self.commands[env_ids, :3] *= (rand_ids < 0.9).unsqueeze(1)

    def _switch(self):
        c_vel = torch.linalg.norm(self.commands, dim=1)
        return torch.exp(-torch.square(torch.max(torch.zeros_like(c_vel),
                                                 c_vel-0.1))/0.1)

    def _reward_swing_grf(self):
        """Reward non-zero grf during swing (0 to pi)"""
        in_contact = \
            torch.gt(torch.norm(self.contact_forces[:, self.feet_indices, :],
                                dim=-1), 50.)
        ph_off = torch.lt(self.phase, torch.pi)
        rew = in_contact*torch.cat((ph_off, ~ph_off, ~ph_off, ph_off), dim=1)
        return -torch.sum(rew.float(), dim=1)*(1-self._switch())

    def _reward_stance_grf(self):
        """Reward non-zero grf during stance (pi to 2pi)"""
        in_contact = torch.gt(torch.norm(
            self.contact_forces[:, self.feet_indices, :], dim=-1), 50.)
        ph_off = torch.gt(self.phase, torch.pi)  # should this be in swing?
        rew = in_contact*torch.cat((ph_off, ~ph_off, ~ph_off, ph_off), dim=1)

        return torch.sum(rew.float(), dim=1)*(1-self._switch())

    def _reward_reference_traj(self):
        """REWARDS EACH LEG INDIVIDUALLY BASED ON ITS POSITION IN THE CYCLE"""
        # * dof position error
        error = self._get_ref() + self.default_dof_pos - self.dof_pos
        error /= self.scales['dof_pos']
        reward = torch.mean(self._sqrdexp(error) - torch.abs(error)*0.2, dim=1)
        # * only when commanded velocity is higher
        return reward*(1-self._switch())

    def _get_ref(self):
        leg_frame = torch.zeros_like(self.torques)
        # offset by half cycle (trot)
        ph_off = torch.fmod(self.phase+torch.pi, 2*torch.pi)
        phd_idx = (torch.round(
            self.phase * (self.leg_ref.size(dim=0)/(2*torch.pi)-1))).long()
        pho_idx = (torch.round(
            ph_off * (self.leg_ref.size(dim=0)/(2*torch.pi)-1))).long()
        leg_frame[:, 0:3] += self.leg_ref[phd_idx.squeeze(), :]
        leg_frame[:, 3:6] += self.leg_ref[pho_idx.squeeze(), :]
        leg_frame[:, 6:9] += self.leg_ref[pho_idx.squeeze(), :]
        leg_frame[:, 9:12] += self.leg_ref[phd_idx.squeeze(), :]
        return leg_frame

    def _reward_stand_still(self):
        """Penalize motion at zero commands"""
        # * normalize angles so we care about being within 5 deg
        rew_pos = torch.mean(self._sqrdexp(
            (self.dof_pos - self.default_dof_pos)/torch.pi*36), dim=1)
        rew_vel = torch.mean(self._sqrdexp(self.dof_vel), dim=1)
        rew_base_vel = torch.mean(torch.square(self.base_lin_vel), dim=1)
        rew_base_vel += torch.mean(torch.square(self.base_ang_vel), dim=1)
        return (rew_vel+rew_pos-rew_base_vel)*self._switch()

    def _reward_tracking_lin_vel(self):
        """Tracking linear velocity commands (xy axes)"""
        # just use lin_vel?
        reward = super()._reward_tracking_lin_vel()
        return reward * (1-self._switch())

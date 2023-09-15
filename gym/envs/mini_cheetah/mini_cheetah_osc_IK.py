import torch
import pandas as pd
from isaacgym.torch_utils import torch_rand_float, to_torch

from gym import LEGGED_GYM_ROOT_DIR
from gym.envs.mini_cheetah.mini_cheetah_osc import MiniCheetahOsc

MINI_CHEETAH_MASS = 8.292 * 9.81  # Weight of mini cheetah in Newtons

class MiniCheetahOscIK(MiniCheetahOsc):

    def _init_buffers(self):
        super()._init_buffers()
        self.ik_pos_target = torch.zeros(self.num_envs, 12,
                                         dtype=torch.float, device=self.device)

    def _pre_physics_step(self):
        super()._pre_physics_step()
        ik_defaults_L = torch.tensor([0., -0.019, -0.5]).to(self.device)
        ik_defaults_R = torch.tensor([0., 0.019, -0.5]).to(self.device)
        joints_ik_fr = self.IK_leg_3DOF(p_hr2ft_hr=self.ik_pos_target[:, 0:3]
                                        + ik_defaults_R)
        joints_ik_fl = self.IK_leg_3DOF(p_hr2ft_hr=self.ik_pos_target[:, 3:6]
                                        + ik_defaults_L)
        joints_ik_br = self.IK_leg_3DOF(p_hr2ft_hr=self.ik_pos_target[:, 6:9]
                                        + ik_defaults_R)
        joints_ik_bl = self.IK_leg_3DOF(p_hr2ft_hr=self.ik_pos_target[:, 9:12]
                                        + ik_defaults_L)
        self.dof_pos_target = torch.cat(
            (joints_ik_fr, joints_ik_fl, joints_ik_br, joints_ik_bl), dim=1)

    def IK_leg_3DOF(self, p_hr2ft_hr):
        # todo: clean the variable names here to make more sense
        r_hr2hp = torch.tensor([0.0, -0.019, 0.0], device=self.device)\
            .unsqueeze(0).repeat(self.num_envs, 1)
        r_hp2kp = torch.tensor([0.0, 0.0, -0.2085], device=self.device)\
            .unsqueeze(0).repeat(self.num_envs, 1)
        r_kp2ft = torch.tensor([0.0, 0.0, -0.22], device=self.device)\
            .unsqueeze(0).repeat(self.num_envs, 1)

        L1 = torch.abs(r_hp2kp[:, 2])
        L2 = torch.abs(r_kp2ft[:, 2])

        # * hip-roll angle
        L_hr2ft_yz = torch.norm(p_hr2ft_hr[:, 1:3], dim=1).clamp(min=1e-6)
        alpha_1 = torch.arcsin(
            torch.clamp(p_hr2ft_hr[:, 1] / L_hr2ft_yz, min=-1, max=1))
        beta_1 = torch.arcsin(
            torch.clamp(torch.abs(r_hr2hp[:, 1])/L_hr2ft_yz, min=-1, max=1))
        q1 = alpha_1 - beta_1

        # * hip-pitch angle
        p_hr2hp_hr = torch.zeros(self.num_envs, 3).to(self.device)
        p_hr2hp_hr[:, 1] = r_hr2hp[:, 1]*torch.cos(q1)
        p_hr2hp_hr[:, 2] = r_hr2hp[:, 1]*torch.sin(q1)
        L_hp2ft_hr = torch.clamp(
            torch.norm(p_hr2ft_hr-p_hr2hp_hr, dim=1), min=1e-6)
        vec = p_hr2ft_hr - p_hr2hp_hr
        p_hp2ft_hp = torch.stack(
            (vec[:, 0],
             vec[:, 1]*torch.cos(q1)+vec[:, 2]*torch.sin(q1),
             vec[:, 2]*torch.cos(q1)-vec[:, 1]*torch.sin(q1)), dim=1)
        alpha_2 = torch.arccos(
            torch.clamp(p_hp2ft_hp[:, 0] / L_hp2ft_hr, min=-1, max=1))
        cos_angle_2 = (L1**2 + L_hp2ft_hr**2 - L2**2)/(2*L1*L_hp2ft_hr)
        cos_angle_2 = torch.clamp(cos_angle_2, min=-1, max=1)
        beta_2 = torch.arccos(cos_angle_2)
        q2 = alpha_2 + beta_2 - torch.pi / 2

        # * knee-pitch angle
        cos_angle_3 = (L1**2+L2**2-L_hp2ft_hr**2)/(2*L1*L2)
        cos_angle_3 = torch.clamp(cos_angle_3, min=-1, max=1)
        acos_angle_3 = torch.arccos(cos_angle_3)
        q3 = -(torch.pi - acos_angle_3)

        return torch.stack((q1, -q2, -q3), dim=1)
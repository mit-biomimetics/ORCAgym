# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import torch
import numpy as np
from isaacgym.torch_utils import (
    get_axis_params, torch_rand_float,
    quat_rotate_inverse, to_torch, quat_from_euler_xyz)
from isaacgym import gymtorch, gymapi

from gym import LEGGED_GYM_ROOT_DIR
from gym.envs.base.base_task import BaseTask
from gym.utils.terrain import Terrain
from gym.utils.math import random_sample, quat_apply_yaw
from gym.utils.helpers import class_to_dict


class LeggedRobot(BaseTask):
    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and
            environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.init_done = False
        self.device = sim_device  # todo CRIME: remove this from __init__ then
        self._parse_cfg(self.cfg)
        super().__init__(gym, sim, self.cfg, sim_params, sim_device, headless)

        if not self.headless:
            self._set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._initialize_sim()
        self.gym.prepare_sim(self.sim)
        self._init_buffers()
        self.init_done = True
        self.reset()

    def step(self):
        """ Apply actions, simulate, call self._post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs,
            num_actions_per_env)
        """
        self._reset_buffers()
        self._pre_physics_step()
        # * step physics and render each frame
        self._render()
        for _ in range(self.cfg.control.decimation):
            self._pre_torque_step()
            self.torques = self._compute_torques()
            self._post_torque_step()
            self._step_physx_sim()

        self._post_physics_step()
        self._check_terminations_and_timeouts()

        env_ids = self.to_be_reset.nonzero().flatten()
        self._reset_idx(env_ids)

    def _pre_physics_step(self):
        return None

    def _pre_torque_step(self):
        return None

    def _post_torque_step(self):
        return None

    def _step_physx_sim(self):
        self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques))
        self.gym.simulate(self.sim)
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)

    def _post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec)

        self.base_height = self.root_states[:, 2:3]

        nact = self.num_actuators
        self.dof_pos_history[:, 2*nact:] = self.dof_pos_history[:, nact:2*nact]
        self.dof_pos_history[:, nact:2*nact] = self.dof_pos_history[:, :nact]
        self.dof_pos_history[:, :nact] = self.dof_pos_target
        self.dof_pos_obs = (self.dof_pos - self.default_dof_pos)

        env_ids = (self.episode_length_buf
                   % int(self.cfg.commands.resampling_time / self.dt) == 0)
        self._resample_commands(env_ids.nonzero().flatten())
        if self.cfg.push_robots.toggle:
            if (self.common_step_counter % self.cfg.push_interval == 0):
                self._push_robots()

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        # * reset robot states
        self._reset_system(env_ids)
        self._resample_commands(env_ids)
        # * reset buffers
        self.dof_pos_history[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

    def _initialize_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are "
                + "[None, plane, heightfield, trimesh]")
        self._create_envs()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]):
            Environments ids for which new commands are needed
        """
        if len(env_ids) == 0:
            return

        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            -self.command_ranges["lin_vel_y"],
            self.command_ranges["lin_vel_y"],
            (len(env_ids), 1),
            device=self.device).squeeze(1)
        max_yaw_vel = self.command_ranges["yaw_vel"]
        self.commands[env_ids, 2] = torch_rand_float(
            -max_yaw_vel, max_yaw_vel, (len(env_ids), 1),
            device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :3], dim=1) > 0.2).unsqueeze(1)

    def _set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape
            properties of each environment. Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]):
                Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]:
                Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0],
                                                    friction_range[1],
                                                    (num_buckets, 1),
                                                    device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of
            each environment. Called During environment creation.
            Base behavior: stores position, velocity and torques limits
                defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2,
                                              dtype=torch.float,
                                              device=self.device)
            self.dof_vel_limits = torch.zeros(self.num_dof,
                                              dtype=torch.float,
                                              device=self.device)
            self.torque_limits = torch.zeros(self.num_dof,
                                             dtype=torch.float,
                                             device=self.device)

            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # * soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = \
                    m - 0.5 * r * self.cfg.reward_settings.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = \
                    m + 0.5 * r * self.cfg.reward_settings.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    def _compute_torques(self):
        if self.cfg.asset.disable_motors:
            self.torques[:] = 0.
            return
        torques = (self.p_gains*(self.dof_pos_target
                                 + self.default_dof_pos
                                 - self.dof_pos)
                   + self.d_gains*(self.dof_vel_target - self.dof_vel)
                   + self.tau_ff)
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
        return torques.view(self.torques.shape)

    def _reset_system(self, env_ids):
        """ Resets selected environmments
        Args:
            env_ids (List[int]): Environemnt ids
        """
        if hasattr(self, self.cfg.init_state.reset_mode):
            eval(f"self.{self.cfg.init_state.reset_mode}(env_ids)")
        else:
            raise NameError(
                f"Unknown default setup: {self.cfg.init_state.reset_mode}")

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32))

        # * start base position shifted in X-Y plane
        if self.custom_origins:
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # * xy position within 1m of the center
            self.root_states[env_ids, :2] += \
                torch_rand_float(
                    -1., 1., (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32))

    # * implement reset methods
    def reset_to_basic(self, env_ids):
        """
        Reset to a single initial state
        """
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0
        self.root_states[env_ids] = self.base_init_state

    def reset_to_range(self, env_ids):
        """
        Reset to a uniformly random distribution of states, sampled from a
        range for each state
        """
        # * dof states
        self.dof_pos[env_ids] = random_sample(
            env_ids, self.dof_pos_range[:, 0],
            self.dof_pos_range[:, 1], device=self.device)
        self.dof_vel[env_ids] = random_sample(
            env_ids, self.dof_vel_range[:, 0],
            self.dof_vel_range[:, 1], device=self.device)

        # * base states
        random_com_pos = random_sample(
            env_ids, self.root_pos_range[:, 0],
            self.root_pos_range[:, 1], device=self.device)

        self.root_states[env_ids, 0:7] = torch.cat(
            (random_com_pos[:, 0:3],
             quat_from_euler_xyz(
                random_com_pos[:, 3],
                random_com_pos[:, 4],
                random_com_pos[:, 5])), 1)
        self.root_states[env_ids, 7:13] = random_sample(
            env_ids, self.root_vel_range[:, 0],
            self.root_vel_range[:, 1], device=self.device)

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a
            randomized base velocity.
        """
        max_vel = self.cfg.push_robots.max_push_vel_xy
        self.root_states[:, 7:9] += torch_rand_float(-max_vel, max_vel,
                                                     (self.num_envs, 2),
                                                     device=self.device)
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and
            processed quantities
        """
        # * get gym GPU state tensors
        actor_root_state = \
            self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = \
            self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = \
            self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = \
            self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # * create some wrapper tensors for different slices

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)

        self._rigid_body_pos = self._rigid_body_state.view(
            self.num_envs, self.num_bodies, 13)[..., 0:3]
        self.dof_pos = self.dof_state.view(
            self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(
            self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        # * shape: num_envs, num_bodies, xyz axis
        self.contact_forces = (
            gymtorch.wrap_tensor(net_contact_forces)
            .view(self.num_envs, -1, 3))

        self._rigid_body_pos = self._rigid_body_state.view(
            self.num_envs, self.num_bodies, 13)[..., 0:3]
        self._rigid_body_quat = self._rigid_body_state.view(
            self.num_envs, self.num_bodies, 13)[..., 3:7]
        self._rigid_body_lin_vel = self._rigid_body_state.view(
            self.num_envs, self.num_bodies, 13)[..., 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state.view(
            self.num_envs, self.num_bodies, 13)[..., 10:13]

        # * initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}

        self.gravity_vec = to_torch(
            get_axis_params(-1., self.up_axis_idx),
            device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actuators,
                                   dtype=torch.float, device=self.device)
        self.p_gains = torch.zeros(self.num_actuators,
                                   dtype=torch.float, device=self.device)
        self.d_gains = torch.zeros(self.num_actuators,
                                   dtype=torch.float, device=self.device)
        self.dof_pos_target = torch.zeros(self.num_envs, self.num_actuators,
                                          dtype=torch.float,
                                          device=self.device)
        self.dof_vel_target = torch.zeros(self.num_envs, self.num_actuators,
                                          dtype=torch.float,
                                          device=self.device)
        self.tau_ff = torch.zeros(self.num_envs, self.num_actuators,
                                  dtype=torch.float, device=self.device)

        self.dof_pos_history = torch.zeros(
            self.num_envs, self.num_actuators*3,
            dtype=torch.float, device=self.device)
        self.commands = torch.zeros(
            self.num_envs, 3,
            dtype=torch.float, device=self.device)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat,
                                                self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat,
                                                self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat,
                                                     self.gravity_vec)
        self.dof_pos_obs = torch.zeros_like(self.dof_pos)
        self.base_height = torch.zeros(self.num_envs, 1,
                                       dtype=torch.float, device=self.device)

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # * joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(
            self.num_dof,
            dtype=torch.float, device=self.device)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angles = self.cfg.init_state.default_joint_angles

            found = False
            for dof_name in angles.keys():
                if dof_name in name:
                    self.default_dof_pos[i] = angles[dof_name]
                    found = True
            if not found:
                self.default_dof_pos[i] = 0.0
                print(f"Default dof pos of joint {name} was not defined, "
                      + "setting to zero")

            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, "
                          + "setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # * check that init range highs and lows are consistent
        # * and repopulate to match
        if self.cfg.init_state.reset_mode == "reset_to_range":
            self.initialize_ranges_for_initial_conditions()

    def initialize_ranges_for_initial_conditions(self):
        self.dof_pos_range = torch.zeros(
            self.num_dof, 2,
            dtype=torch.float, device=self.device)
        self.dof_vel_range = torch.zeros(
            self.num_dof, 2,
            dtype=torch.float, device=self.device)

        for joint, vals in self.cfg.init_state.dof_pos_range.items():
            for i in range(self.num_dof):
                if joint in self.dof_names[i]:
                    self.dof_pos_range[i, :] = to_torch(
                        vals, device=self.device)

        for joint, vals in self.cfg.init_state.dof_vel_range.items():
            for i in range(self.num_dof):
                if joint in self.dof_names[i]:
                    self.dof_vel_range[i, :] = to_torch(
                        vals, device=self.device)

        self.root_pos_range = torch.tensor(
            self.cfg.init_state.root_pos_range,
            dtype=torch.float, device=self.device)
        self.root_vel_range = torch.tensor(
            self.cfg.init_state.root_vel_range,
            dtype=torch.float, device=self.device)

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and
            restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters
            based on the cfg.
        """
        hf_params = gymapi.HeightFieldProperties()
        hf_params.column_scale = self.terrain.horizontal_scale
        hf_params.row_scale = self.terrain.horizontal_scale
        hf_params.vertical_scale = self.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.border_size
        hf_params.transform.p.y = -self.terrain.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(
            self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(
            self.terrain.heightsamples).view(
                self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters
            based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(
            self.sim, self.terrain.vertices.flatten(order='C'),
            self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(
            self.terrain.heightsamples).view(
                self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        asset_options.default_dof_drive_mode = \
            self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = \
            self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = \
            self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = \
            self.cfg.asset.flip_visual_attachments
        asset_options.max_angular_velocity = \
            self.cfg.asset.max_angular_velocity

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        dof_props_asset['armature'] = self.cfg.asset.rotor_inertia
        dof_props_asset['damping'] = self.cfg.asset.joint_damping
        rigid_shape_props_asset = \
            self.gym.get_asset_rigid_shape_properties(robot_asset)

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend(
                [s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend(
                [s for s in body_names if name in s])

        base_init_state_list = (
            self.cfg.init_state.pos + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel)
        self.base_init_state = (
            to_torch(base_init_state_list,
                     device=self.device))
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # * create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper,
                                             int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1),
                                        device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = \
                self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset,
                                                      rigid_shape_props)

            name = self.cfg.asset.file
            name = name.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            robot_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, name, i,
                self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(
                env_handle, robot_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, robot_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, robot_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(robot_handle)
        self.feet_indices = torch.zeros(
            len(feet_names),
            dtype=torch.long, device=self.device)
        for i in range(len(feet_names)):
            self.feet_indices[i] = \
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long, device=self.device)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = \
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0],
                    penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long, device=self.device)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = \
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0],
                    termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined
            by the terrain platforms. Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(
                self.num_envs, 3,
                device=self.device)
            # * put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(
                0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(
                torch.arange(self.num_envs, device=self.device),
                (self.num_envs/self.cfg.terrain.num_cols),
                rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(
                self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = \
                self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(
                self.num_envs, 3,
                device=self.device)
            # * create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(
                torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.ctrl_dt
        self.scales = class_to_dict(self.cfg.scaling, self.device)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = int(np.ceil(
            self.max_episode_length_s / self.dt))
        self.cfg.push_interval = np.ceil(
            self.cfg.push_robots.interval_s / self.dt)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled
            (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape
                (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(
            self.cfg.terrain.measured_points_y,
            device=self.device)
        x = torch.tensor(
            self.cfg.terrain.measured_points_x,
            device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(
            self.num_envs, self.num_height_points, 3,
            device=self.device)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each
            robot. The points are offset by the base's position and rotated
            by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to
                return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(
                self.num_envs, self.num_height_points,
                device=self.device)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError(
                "Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_height_points),
                (self.height_points[env_ids])
                + (self.root_states[env_ids, :3]).unsqueeze(1))
        else:
            points = quat_apply_yaw(
                self.base_quat.repeat(1, self.num_height_points),
                self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) \
            * self.terrain.cfg.vertical_scale

    def _sqrdexp(self, x, scale=None):
        """ shorthand helper for squared exponential
        """
        if scale is None:
            scale = 1.
        return torch.exp(
            -torch.square(x/scale)/self.cfg.reward_settings.tracking_sigma)

    # ------------ reward functions----------------

    def _reward_lin_vel_z(self):
        """Penalize z axis base linear velocity"""
        return -torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        """Penalize xy axes base angular velocity"""
        return -torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        """Penalize non flat base orientation"""
        return -torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        """Penalize base height away from target"""
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return -torch.square(
            base_height - self.cfg.reward_settings.base_height_target)

    def _reward_torques(self):
        """Penalize torques"""
        return -torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """Penalize dof velocities"""
        return -torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_action_rate(self):
        """Penalize changes in actions"""
        nact = self.num_actuators
        dt2 = (self.dt*self.cfg.control.decimation)**2
        error = torch.square(self.dof_pos_history[:, :nact]
                             - self.dof_pos_history[:, nact:2*nact])/dt2
        return -torch.sum(error, dim=1)

    def _reward_action_rate2(self):
        """Penalize changes in actions"""
        nact = self.num_actuators
        dt2 = (self.dt*self.cfg.control.decimation)**2
        error = torch.square(self.dof_pos_history[:, :nact]
                             - 2*self.dof_pos_history[:, nact:2*nact]
                             + self.dof_pos_history[:, 2*nact:])/dt2
        return -torch.sum(error, dim=1)

    def _reward_collision(self):
        """Penalize collisions on selected bodies"""
        return -torch.sum(1.*(torch.norm(
            self.contact_forces[:, self.penalised_contact_indices, :],
            dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return -self.terminated.float()

    def _reward_dof_pos_limits(self):
        """Penalize dof positions too close to the limit"""
        # * lower limit
        out_of_limits = -(
            self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (
            self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return -torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        """Penalize dof velocities too close to the limit"""
        # * clip to max error = 1 rad/s per joint to avoid huge penalties
        soft_dof_vel_limit = self.cfg.reward_settings.soft_dof_vel_limit
        return -torch.sum((
            torch.abs(self.dof_vel)
            - (self.dof_vel_limits * soft_dof_vel_limit))
                         .clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        """penalize torques too close to the limit"""
        return -torch.sum((
            torch.abs(self.torques)
            - self.torque_limits*self.cfg.reward_settings.soft_torque_limit)
                         .clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        """Tracking of linear velocity commands (xy axes)"""
        error = torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2])
        error = torch.exp(-error/self.cfg.reward_settings.tracking_sigma)
        return torch.sum(error, dim=1)

    def _reward_tracking_ang_vel(self):
        """Tracking of angular velocity commands (yaw)"""
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return self._sqrdexp(ang_vel_error/self.scales['base_ang_vel'][2])

    def _reward_feet_contact_forces(self):
        """penalize high contact forces"""
        return -torch.sum((
            torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
            - self.cfg.reward_settings.max_contact_force).clip(min=0.), dim=1)

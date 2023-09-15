import os
import torch
import numpy as np
from isaacgym.torch_utils import to_torch, get_axis_params
from isaacgym import gymtorch, gymapi

from gym import LEGGED_GYM_ROOT_DIR
from gym.envs.base.base_task import BaseTask
from gym.utils.math import random_sample, exp_avg_filter
from gym.utils.helpers import class_to_dict


class FixedRobot(BaseTask):

    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain
            and environments), initilizes pytorch buffers used during training

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
        self.init_done = False
        self.device = sim_device
        self._parse_cfg(self.cfg)
        super().__init__(gym, sim, self.cfg, sim_params,
                         sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._initialize_sim()
        self.gym.prepare_sim(self.sim)
        self._init_buffers()
        self.init_done = True
        self.reset()

    def step(self):
        """ Apply actions, simulate, call self.post_physics_step()
            and pre_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape
                (num_envs, num_actions_per_env)
        """

        self._reset_buffers()
        self._pre_physics_step()
        # * step physics and render each frame
        self._render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques()

            if self.cfg.asset.disable_motors:
                self.torques[:] = 0.
            torques_to_gym_tensor = torch.zeros(self.num_envs, self.num_dof,
                                                device=self.device)

            # todo encapsulate
            next_torques_idx = 0
            for dof_idx in range(self.num_dof):
                if self.cfg.control.actuated_joints_mask[dof_idx]:
                    torques_to_gym_tensor[:, dof_idx] = \
                        self.torques[:, next_torques_idx]
                    next_torques_idx += 1
                else:
                    torques_to_gym_tensor[:, dof_idx] = torch.zeros(
                        self.num_envs, device=self.device)

            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(torques_to_gym_tensor))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        self._post_physics_step()
        self._check_terminations_and_timeouts()

        env_ids = self.to_be_reset.nonzero().flatten()
        self._reset_idx(env_ids)

    def _pre_physics_step(self):
        pass

    def _post_physics_step(self):
        """
        check terminations, compute observations and rewards
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        nact = self.num_actuators
        self.dof_pos_history[:, 2*nact:] = self.dof_pos_history[:, nact:2*nact]
        self.dof_pos_history[:, nact:2*nact] = self.dof_pos_history[:, :nact]
        self.dof_pos_history[:, :nact] = self.dof_pos_target

        self.dof_pos_obs = self.dof_pos - self.default_dof_pos

    def _reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids),
            and self._resample_commands(env_ids)
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        # * reset robot states
        self._reset_system(env_ids)
        # * reset buffers
        self.dof_pos_history[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

    def _initialize_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self._create_ground_plane()
        self._create_envs()

    def set_camera(self, position, lookat):
        """
        Set the camera position and lookat.
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape
            properties of each environment. Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each
                shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape
                properties
        """
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
        if env_id == 0:  # ? why only for env_id == 0?
            self.dof_pos_limits = torch.zeros(self.num_dof, 2,
                                              dtype=torch.float,
                                              device=self.device)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float,
                                              device=self.device)
            self.torque_limits = torch.zeros(self.num_actuators,
                                             dtype=torch.float,
                                             device=self.device)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                try:  # ! this seems like a crime...
                    self.torque_limits[i] = props["effort"][i].item()
                except:
                    print("WARNING: passive joints need to be listed after "
                          + "active joints in the URDF.")
                # * soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = \
                    m - 0.5 * r * self.cfg.reward_settings.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = \
                    m + 0.5 * r * self.cfg.reward_settings.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        """
        Process rigid body properties.
        In `legged_robot` this is used to randomize the base mass.
        Implement as you see fit.
        """
        return props

    def _compute_torques(self):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given
            to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs,
                even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        actuated_dof_pos = torch.zeros(self.num_envs, self.num_actuators,
                                       device=self.device)
        actuated_dof_vel = torch.zeros(self.num_envs, self.num_actuators,
                                       device=self.device)
        for dof_idx in range(self.num_dof):
            idx = 0
            if self.cfg.control.actuated_joints_mask[dof_idx]:
                actuated_dof_pos[:, idx] = self.dof_pos[:, dof_idx]
                actuated_dof_vel[:, idx] = self.dof_vel[:, dof_idx]
                idx += 1

        torques = (
            self.p_gains * (self.dof_pos_target + self.default_act_pos
                            - actuated_dof_pos)
            + self.d_gains * (self.dof_vel_target - actuated_dof_vel)
            + self.tau_ff)
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
        return torques.view(self.torques.shape)

    def _reset_system(self, env_ids):
        """
        Reset the system.
        """

        if hasattr(self, self.cfg.init_state.reset_mode):
            eval(f"self.{self.cfg.init_state.reset_mode}(env_ids)")
        else:
            raise NameError(
                f"Unknown default setup: {self.cfg.init_state.reset_mode}")

        # self.root_states[env_ids] = self.base_init_state
        # self.root_states[env_ids, 7:13] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # * implement reset methods
    def reset_to_basic(self, env_ids):
        """
        Generate random samples for each entry of env_ids
        todo: pass in the actual number instead of the list env_ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0

    def reset_to_range(self, env_ids):
        """
        Reset to a uniformly random distribution of states, sampled from a
        range for each state
        """
        # dof states
        self.dof_pos[env_ids] = random_sample(
            env_ids, self.dof_pos_range[:, 0],
            self.dof_pos_range[:, 1], device=self.device)
        self.dof_vel[env_ids] = random_sample(
            env_ids, self.dof_vel_range[:, 0],
            self.dof_vel_range[:, 1], device=self.device)

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and
        processed quantities
        """
        n_envs = self.num_envs
        # * get gym GPU state tensors
        actor_root_state = \
            self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = \
            self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = \
            self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # * create some wrapper tensors for different slices
        # ! root_states probably not needed...
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.dof_pos = self.dof_state.view(n_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(n_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        # * shape: num_envs, num_bodies, xyz axis
        self.contact_forces = \
            gymtorch.wrap_tensor(net_contact_forces).view(n_envs, -1, 3)

        # * initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx),
                                    device=self.device).repeat((n_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.],
                                    device=self.device).repeat((n_envs, 1))
        self.torques = torch.zeros(n_envs, self.num_actuators,
                                   dtype=torch.float, device=self.device)
        self.p_gains = torch.zeros(self.num_actuators,
                                   dtype=torch.float, device=self.device)
        self.d_gains = torch.zeros(self.num_actuators,
                                   dtype=torch.float, device=self.device)
        self.actions = torch.zeros(n_envs, self.num_actuators,
                                   dtype=torch.float, device=self.device)
        self.dof_pos_target = torch.zeros(self.num_envs, self.num_actuators,
                                          dtype=torch.float,
                                          device=self.device)
        self.dof_vel_target = torch.zeros(self.num_envs,
                                          self.num_actuators,
                                          dtype=torch.float,
                                          device=self.device)
        self.tau_ff = torch.zeros(self.num_envs, self.num_actuators,
                                  dtype=torch.float, device=self.device)
        self.dof_pos_history = torch.zeros(self.num_envs,
                                           self.num_actuators*3,
                                           dtype=torch.float,
                                           device=self.device)
        # * joint positions offsets and PD gains
        # * added: default_act_pos, to differentiate from passive joints
        self.default_dof_pos = torch.zeros(self.num_dof,
                                           dtype=torch.float,
                                           device=self.device)
        self.default_act_pos = torch.zeros(self.num_actuators,
                                           dtype=torch.float,
                                           device=self.device)
        actuated_idx = []  # temp
        for i in range(self.num_dof):
            name = self.dof_names[i]
            # angle = self.cfg.init_state.default_joint_angles[name]
            angles = self.cfg.init_state.default_joint_angles
            # self.default_dof_pos[i] = angle
            found = False
            for dof_name in angles.keys():
                if dof_name in name:
                    self.default_dof_pos[i] = angles[dof_name]
                    found = True
            if not found:
                self.default_dof_pos[i] = 0.
                print(f"Default dof pos of joint {name} was not defined, "
                      + "setting to zero")

            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    self.default_act_pos[i] = angles[dof_name]
                    found = True
                    actuated_idx.append(i)
            if not found:
                try:
                    self.p_gains[i] = 0.
                    self.d_gains[i] = 0.
                    # todo remove if unnecessary
                    print("This should not happen anymore")
                    if self.cfg.control.control_type in ["P", "V"]:
                        print(f"PD gain of joint {name} not defined, "
                              + "set to zero")
                except:  # ! another crime - no bare excepts
                    pass

        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_act_pos = self.default_act_pos.unsqueeze(0)
        # * store indices of actuated joints
        self.act_idx = to_torch(actuated_idx, dtype=torch.long,
                                device=self.device)
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
                    self.dof_pos_range[i, :] = to_torch(vals,
                                                        device=self.device)

        for joint, vals in self.cfg.init_state.dof_vel_range.items():
            for i in range(self.num_dof):
                if joint in self.dof_names[i]:
                    self.dof_vel_range[i, :] = to_torch(vals,
                                                        device=self.device)

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
        asset_options.disable_gravity = \
            self.cfg.asset.disable_gravity
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

        # * save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend(
                [s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend(
                [s for s in body_names if name in s])

        start_pose = gymapi.Transform()

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)  # ? what's this?
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(
                rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(
                robot_asset, rigid_shape_props)
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

    def _get_env_origins(self):  # TODO: do without terrain
        """ Sets environment origins. On rough terrain the origins are defined
            by the terrain platforms. Otherwise create a grid.
        """
        # * removed terrain options
        self.custom_origins = False
        self.env_origins = torch.zeros(
            self.num_envs, 3,
            device=self.device)
        # * create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = self.cfg.env.root_height

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.ctrl_dt
        self.scales = class_to_dict(self.cfg.scaling, self.device)
        # self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

    # ------------ reward functions----------------

    def _sqrdexp(self, x):
        """ shorthand helper for squared exponential
        """
        return torch.exp(
            -torch.square(x)/self.cfg.reward_settings.tracking_sigma)

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
        error = torch.square(
            self.dof_pos_history[:, :nact]
            - self.dof_pos_history[:, nact:2*nact])/dt2
        return -torch.sum(error, dim=1)

    def _reward_action_rate2(self):
        """Penalize changes in actions"""
        nact = self.num_actuators
        dt2 = (self.dt*self.cfg.control.decimation)**2
        error = torch.square(
            self.dof_pos_history[:, :nact]
            - 2*self.dof_pos_history[:, nact:2*nact]
            + self.dof_pos_history[:, 2*nact:])/dt2
        return -torch.sum(error, dim=1)

    def _reward_collision(self):
        """Penalize collisions on selected bodies"""
        return -torch.sum(
            1.*(torch.norm(
                self.contact_forces[:, self.penalised_contact_indices, :],
                dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return -self.terminated.float()

    def _reward_dof_pos_limits(self):
        """Penalize dof positions too close to the limit"""
        out_of_limits = (
            -(self.dof_pos - self.dof_pos_limits[:, 0])
            .clip(max=0.))  # lower limit
        out_of_limits += (
            self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return -torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        """Penalize dof velocities too close to the limit"""
        # * clip to max error = 1 rad/s per joint to avoid huge penalties
        return -torch.sum(
            (torch.abs(self.dof_vel)
             - self.dof_vel_limits*self.cfg.reward_settings.soft_dof_vel_limit)
            .clip(min=0., max=1.), dim=1)

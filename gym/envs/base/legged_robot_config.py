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

from .base_config import BaseConfig


class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_actuators = 12
        env_spacing = 0.  # not used with heightfields/trimeshes
        episode_length_s = 20  # episode length in seconds

    class terrain:
        mesh_type = 'plane'  # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # * rough terrain only:
        measure_heights = False
        # * 1mx1.6m rectangle (without center line)
        measured_points_x = [
            -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
            0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [
            -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 2  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # * terrain types:
        # * [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.6, 0.4, 0., 0., 0.]
        # * trimesh only:
        slope_treshold = 0.75
        # * slopes above this threshold will be corrected to vertical surfaces

    class commands:
        resampling_time = 10.  # time before command are changed[s]

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = 1.  # max [m/s]
            yaw_vel = 1    # max [rad/s]

    class push_robots:
        toggle = True
        interval_s = 15
        max_push_vel_xy = 0.05

    class init_state:

        # * target state when action = 0, also reset positions for basic mode
        default_joint_angles = {"joint_a": 0.,
                                "joint_b": 0.}

        # * reset setup chooses how the initial conditions are chosen.
        # * "reset_to_basic" = a single position
        # * "reset_to_range" = uniformly random from a range defined below
        reset_mode = "reset_to_basic"

        # * default COM for basic initialization
        pos = [0.0, 0.0, 1.]        # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]   # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]   # x,y,z [rad/s]

        # * initial conditiosn for reset_to_range
        dof_pos_range = {'joint_a': [-1., 1.],
                         'joint_b': [-1., 1.]}
        dof_vel_range = {'joint_a': [-1., 1.],
                         'joint_b': [-1., 1.]}
        root_pos_range = [[0., 0.],     # x
                          [0., 0.],     # y
                          [0.5, 0.75],  # z
                          [0., 0.],     # roll
                          [0., 0.],     # pitch
                          [0., 0.]]     # yaw
        root_vel_range = [[-0.1, 0.1],  # x
                          [-0.1, 0.1],  # y
                          [-0.1, 0.1],  # z
                          [-0.1, 0.1],  # roll
                          [-0.1, 0.1],  # pitch
                          [-0.1, 0.1]]  # yaw

    class control:
        # * PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        ctrl_frequency = 100
        desired_sim_frequency = 200

    class asset:
        file = ""
        # * name of the feet bodies,
        # * used to index body state and contact force tensors
        foot_name = "None"
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        end_effector_names = []
        disable_gravity = False
        disable_motors = False
        # * merge bodies connected by fixed joints.
        # * Specific fixed joints can be kept by adding
        # * " <... dont_collapse="true">
        collapse_fixed_joints = True
        # * fix the base of the robot
        fix_base_link = False
        # * see GymDofDriveModeFlags
        # * (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        default_dof_drive_mode = 3
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True
        # * Some .obj meshes must be flipped from y-up to z-up
        flip_visual_attachments = True

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01
        rotor_inertia = 0.
        joint_damping = 0.

    class domain_rand:
        randomize_friction = False
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]  # added to each link!

    class reward_settings:
        # * tracking reward = exp(-error^2/sigma)
        tracking_sigma = 0.25
        # * percentage of urdf limits, values above this limit are penalized
        soft_dof_pos_limit = 1.
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        # * forces above this value are penalized
        max_contact_force = 100.

    class scaling:
        commands = 1
        # * Action scales
        tau_ff = 1  # scale by torque limits
        dof_pos = 1
        dof_pos_obs = dof_pos
        dof_pos_target = dof_pos  # scale by range of motion

    # * viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]


class LeggedRobotRunnerCfg(BaseConfig):
    seed = -1
    runner_class_name = 'OnPolicyRunner'

    class logging():
        enable_local_saving = True

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = 'elu'

        actor_obs = [
            "observation_a",
            "observation_b",
            "these_need_to_be_atributes_(states)_of_the_robot_env"]
        critic_obs = [
            "observation_x",
            "observation_y",
            "critic_obs_can_be_the_same_or_different_than_actor_obs"]

        actions = ["q_des"]
        disable_actions = False

        class noise:
            dof_pos_obs = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            projected_gravity = 0.05
            height_measurements = 0.1

        class reward:
            class weights:
                tracking_lin_vel = .0
                tracking_ang_vel = 0.
                lin_vel_z = 0
                ang_vel_xy = 0.
                orientation = 0.
                torques = 0.
                dof_vel = 0.
                base_height = 0.
                collision = 0.
                action_rate = 0.
                action_rate2 = 0.
                stand_still = 0.
                dof_pos_limits = 0.

            class termination_weight:
                termination = 0.01

    class algorithm:
        # * training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        # * mini batch size = num_envs*nsteps / nminibatches
        num_mini_batches = 4
        learning_rate = 1.e-3
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 1500
        save_interval = 50
        run_name = ''
        experiment_name = 'legged_robot'

        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        device = 'cpu'

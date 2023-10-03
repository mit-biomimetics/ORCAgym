from gym.envs.mini_cheetah.mini_cheetah_config \
    import MiniCheetahCfg, MiniCheetahRunnerCfg

BASE_HEIGHT_REF = 0.33


class MiniCheetahOscCfg(MiniCheetahCfg):

    class viewer:
        ref_env = 0
        pos = [6, 10, 4]  # [m]
        lookat = [6, 0, 1]  # [m]

    class env(MiniCheetahCfg.env):
        num_envs = 4096
        num_actuators = 12
        episode_length_s = 4.
        env_spacing = 3.

    class terrain(MiniCheetahCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh'  # none, plane, heightfield or trimesh

    class init_state(MiniCheetahCfg.init_state):
        reset_mode = "reset_to_range"
        timeout_reset_ratio = 0.75
        # * default COM for basic initialization 
        pos = [0.0, 0.0, 0.35]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {
            "haa": 0.0,
            "hfe": -0.785398,
            "kfe": 1.596976,
        }

        # * initialization for random range setup
        dof_pos_range = {'haa': [-0.00001, 0.00001],
                         'hfe': [-0.785398, -0.785398],
                         'kfe': [1.596976, 1.596976]}
        dof_vel_range = {'haa': [0., 0.],
                         'hfe': [0., 0.],
                         'kfe': [0., 0.]}
        root_pos_range = [[0., 0.],       # x
                          [0., 0.],       # y
                          [0.35, 0.35],    # z
                          [0., 0.],       # roll
                          [0., 0.],       # pitch
                          [0., 0.]]       # yaw
        root_vel_range = [[-0.5, 2.],  # x
                          [0., 0.],       # y
                          [-0.05, 0.05],  # z
                          [0., 0.],       # roll
                          [0., 0.],       # pitch
                          [0., 0.]]       # yaw

    class control(MiniCheetahCfg.control):
        # * PD Drive parameters:
        stiffness = {'haa': 20., 'hfe': 20., 'kfe': 20.}
        damping = {'haa': 0.5, 'hfe': 0.5, 'kfe': 0.5}
        ctrl_frequency = 100
        desired_sim_frequency = 500

    class osc:
        process_noise_std = 0.25
        grf_threshold = 0.1  # 20. # Normalized to body weight
        # oscillator parameters
        omega = 3  # gets overwritten
        coupling = 1  # gets overwritten
        osc_bool = False  # not used in paper
        grf_bool = False  # not used in paper
        randomize_osc_params = False
        # randomization ranges: not used in paper
        omega_range = [1.0, 4.0]  # [0.0, 10.]
        coupling_range = [0.0, 1.0]  # with normalized grf, can have omega/coupling on same scale
        offset_range = [0.0, 0.0]

        # choice of oscillator parameters, see paper equation (5)
        stop_threshold = 0.5
        omega_stop = 1.
        omega_step = 2.
        omega_slope = 1.
        omega_max = 4.0
        omega_var = 0.25
        coupling_stop = 4.
        coupling_step = 1.
        coupling_slope = 0.
        coupling_max = 1.0
        offset = 1.
        coupling_var = 0.25

        init_to = 'random'
        init_w_offset = True


    class commands:
        resampling_time = 3.  # time before command are changed[s]
        var = 1.0
        class ranges:
            lin_vel_x = [-3., -1., 0., 1., 3.]  # min max [m/s]
            lin_vel_y = 1.  # [-1., 0, 1.]  # max [m/s]
            yaw_vel = 3.  # [-6., -3., 0., 3., 6.]    # max [rad/s]

    class push_robots:
        toggle = True
        interval_s = 10
        max_push_vel_xy = 0.2

    class domain_rand(MiniCheetahCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.4, 1.0]
        randomize_base_mass = False
        lower_mass_offset = -0.5  # kg
        upper_mass_offset = 2.
        lower_z_offset = 0.0  # m
        upper_z_offset = 0.2
        lower_x_offset = 0.0
        upper_x_offset = 0.0

    class asset(MiniCheetahCfg.asset):
        shank_length_diff = 0  # Units in cm
        # file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/" \
        #     + "mini_cheetah/urdf/mini_cheetah_" \
        #     + str(shank_length_diff) + ".urdf"
        file = ("{LEGGED_GYM_ROOT_DIR}/resources/robots/"
                + "mini_cheetah/urdf/mini_cheetah_0.urdf")
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "shank"]
        terminate_after_contacts_on = ["base"]
        collapse_fixed_joints = False
        fix_base_link = False
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        disable_gravity = False
        disable_motors = False  # all torques set to 0

    class reward_settings(MiniCheetahCfg.reward_settings):
        soft_dof_pos_limit = 0.8
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 600.
        base_height_target = BASE_HEIGHT_REF+0.03
        tracking_sigma = 0.25
        switch_scale = 0.5  # 0.1

    class scaling(MiniCheetahCfg.scaling):
        base_ang_vel = [0.3, 0.3, 0.1]
        base_lin_vel = BASE_HEIGHT_REF
        # dof_vel = 100.
        # dof_vel = 4*[41, 41, 27]  # ought to be roughly max expected speed.
        dof_vel = 4*[2., 2., 4.]
        base_height = 0.3
        dof_pos = 4*[0.2, 0.3, 0.3]  # hip-abad, hip-pitch, knee
        dof_pos_obs = dof_pos
        dof_pos_target = 4*[0.2, 0.3, 0.3]
        tau_ff = 4*[18, 18, 28]  # hip-abad, hip-pitch, knee
        # commands = [base_lin_vel, base_lin_vel, base_ang_vel]
        commands = [3, 1, 3]  # [base_lin_vel, base_lin_vel, base_ang_vel]

class MiniCheetahOscRunnerCfg(MiniCheetahRunnerCfg):
    seed = -1

    class policy(MiniCheetahRunnerCfg.policy):
        actor_hidden_dims = [256, 256, 128]
        critic_hidden_dims = [256, 256, 128]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = 'elu'

        actor_obs = [
                     "base_ang_vel",
                     "projected_gravity",
                     "commands",
                     "dof_pos_obs",
                     "dof_vel",
                     "oscillator_obs",
                     "dof_pos_target",
                    #  "osc_omega",
                    #  "osc_coupling"
                     #  "oscillators_vel",
                     #  "grf",
                     ]

        critic_obs = ["base_height",
                      "base_lin_vel",
                      "base_ang_vel",
                      "projected_gravity",
                      "commands",
                      "dof_pos_obs",
                      "dof_vel",
                      "oscillator_obs",
                      "oscillators_vel",
                      "dof_pos_target",
                    #   "osc_omega",
                    #   "osc_coupling"
                    #   "osc_omega",
                    #   "osc_coupling",
                    #   "osc_offset"
                      ]

        actions = ["dof_pos_target"]

        class noise:
            scale = 1.0
            dof_pos_obs = 0.01  # can be made very low
            base_ang_vel = 0.01
            dof_pos = 0.005
            dof_vel = 0.005
            lin_vel = 0.05
            ang_vel = [0.3, 0.15, 0.4]
            gravity_vec = 0.1

        class reward:
            class weights:
                tracking_lin_vel = 4.0
                tracking_ang_vel = 2.0
                lin_vel_z = 0.
                ang_vel_xy = 0.
                orientation = 1.0
                torques = 5.e-7
                dof_vel = 0.
                min_base_height = 1.5
                collision = 0
                action_rate = 0.01  # -0.01
                action_rate2 = 0.001  # -0.001
                stand_still = 0.
                dof_pos_limits = 0.
                feet_contact_forces = 0.
                dof_near_home = 0.
                swing_grf = 5.
                stance_grf = 5.
                swing_velocity = 0.
                stance_velocity = 0.
                coupled_grf = 0.  # 8.
                enc_pace = 0.
                cursorial = 0.25
                standing_torques = 0.  # 1.e-5

            class termination_weight:
                termination = 15./100.

    class algorithm(MiniCheetahRunnerCfg.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.02
        num_learning_epochs = 4
        # mini batch size = num_envs*nsteps/nminibatches
        num_mini_batches = 8
        learning_rate = 1.e-5
        schedule = 'adaptive'  # can be adaptive, fixed
        discount_horizon = 1.
        GAE_bootstrap_horizon = 2.
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner(MiniCheetahRunnerCfg.runner):
        run_name = ''
        experiment_name = 'FullSend'
        max_iterations = 2000  # number of policy updates
        algorithm_class_name = 'PPO'
        num_steps_per_env = 32

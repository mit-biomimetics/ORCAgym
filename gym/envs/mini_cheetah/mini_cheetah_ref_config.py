from gym.envs.mini_cheetah.mini_cheetah_config \
    import MiniCheetahCfg, MiniCheetahRunnerCfg

BASE_HEIGHT_REF = 0.33


class MiniCheetahRefCfg(MiniCheetahCfg):
    class env(MiniCheetahCfg.env):
        num_envs = 4096
        num_actuators = 12
        episode_length_s = 15.

    class terrain(MiniCheetahCfg.terrain):
        pass

    class init_state(MiniCheetahCfg.init_state):
        reset_mode = "reset_to_basic"
        # * default COM for basic initialization
        pos = [0.0, 0.0, 0.33]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {
            "haa": 0.0,
            "hfe": -0.785398,
            "kfe": 1.596976,
        }
        # * range
        # * initialization for random range setup
        dof_pos_range = {'haa': [-0.05, 0.05],
                         'hfe': [-0.85, -0.6],
                         'kfe': [-1.45, 1.72]}
        dof_vel_range = {'haa': [0., 0.],
                         'hfe': [0., 0.],
                         'kfe': [0., 0.]}
        root_pos_range = [[0., 0.],       # x
                          [0., 0.],       # y
                          [0.35, 0.4],    # z
                          [0., 0.],       # roll
                          [0., 0.],       # pitch
                          [0., 0.]]       # yaw
        root_vel_range = [[-0.5, 2.],  # x
                          [0., 0.],       # y
                          [-0.05, 0.05],  # z
                          [0., 0.],       # roll
                          [0., 0.],       # pitch
                          [0., 0.]]       # yaw

        ref_traj = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/"
            + "mini_cheetah/trajectories/single_leg.csv")

    class control(MiniCheetahCfg.control):
        # * PD Drive parameters:
        stiffness = {'haa': 20., 'hfe': 20., 'kfe': 20.}
        damping = {'haa': 0.5, 'hfe': 0.5, 'kfe': 0.5}
        gait_freq = 4.
        ctrl_frequency = 100
        desired_sim_frequency = 1000

    class commands:
        resampling_time = 4.  # time before command are changed[s]

        class ranges:
            lin_vel_x = [0., 3.]  # min max [m/s]
            lin_vel_y = 1.  # max [m/s]
            yaw_vel = 3.14/2.    # max [rad/s]

    class push_robots:
        toggle = True
        interval_s = 10
        max_push_vel_xy = 0.2

    class domain_rand(MiniCheetahCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.75, 1.05]  # ! zap, replace with the one below
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        friction_range = [0., 1.0]

    class asset(MiniCheetahCfg.asset):
        file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/"
            + "mini_cheetah/urdf/mini_cheetah_rotor.urdf")
        foot_name = "foot"
        penalize_contacts_on = ["shank"]
        terminate_after_contacts_on = ["base", "thigh"]
        collapse_fixed_joints = False
        fix_base_link = False
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        disable_gravity = False
        disable_motors = False  # all torques set to 0

    class reward_settings(MiniCheetahCfg.reward_settings):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 600.
        base_height_target = BASE_HEIGHT_REF
        tracking_sigma = 0.3

    class scaling(MiniCheetahCfg.scaling):
        base_ang_vel = 3.14/(BASE_HEIGHT_REF/9.81)**0.5
        base_lin_vel = BASE_HEIGHT_REF
        dof_vel = 100.  # ought to be roughly max expected speed.
        base_height = BASE_HEIGHT_REF
        dof_pos = 4*[0.1, 1., 2]  # hip-abad, hip-pitch, knee
        dof_pos_obs = dof_pos
        dof_pos_target = 0.75  # action_scale
        tau_ff = 4*[18, 18, 28]  # hip-abad, hip-pitch, knee
        commands = [base_lin_vel, base_lin_vel, base_ang_vel]


class MiniCheetahRefRunnerCfg(MiniCheetahRunnerCfg):
    seed = -1

    class policy(MiniCheetahRunnerCfg.policy):
        actor_hidden_dims = [256, 256, 128]
        critic_hidden_dims = [256, 256, 128]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = 'elu'

        actor_obs = ["base_height",
                     "base_lin_vel",
                     "base_ang_vel",
                     "projected_gravity",
                     "commands",
                     "dof_pos_obs",
                     "dof_vel",
                     "phase_obs"
                     ]

        critic_obs = actor_obs

        actions = ["dof_pos_target"]

        class noise:
            dof_pos_obs = 0.  # 0.005  # can be made very low
            dof_vel = 0.  # 0.005
            ang_vel = 0.  # [0.1, 0.1, 0.1]  # 0.027, 0.14, 0.37
            base_ang_vel = 0.  # 0.
            dof_pos = 0.  # 0.005
            dof_vel = 0.  # 0.005
            lin_vel = 0.  # 0.
            ang_vel = 0.  # [0.3, 0.15, 0.4]
            gravity_vec = 0.  # 0.05

        class reward:
            class weights:
                tracking_lin_vel = 4.0
                tracking_ang_vel = 1.0
                lin_vel_z = 0.
                ang_vel_xy = 0.
                orientation = 1.75
                torques = 5.e-7
                dof_vel = 0.
                min_base_height = 1.5
                collision = 0.25
                action_rate = 0.01
                action_rate2 = 0.001
                stand_still = 0.
                dof_pos_limits = 0.
                feet_contact_forces = 0.
                dof_near_home = 0.
                reference_traj = 0.5
                swing_grf = 3.
                stance_grf = 3.

            class termination_weight:
                termination = 0.15

    class algorithm(MiniCheetahRunnerCfg.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 6
        # mini batch size = num_envs*nsteps/nminibatches
        num_mini_batches = 6
        learning_rate = 5.e-5
        schedule = 'adaptive'  # can be adaptive, fixed
        discount_horizon = 1.  # [s]
        GAE_bootstrap_horizon = 1.  # [s]
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner(MiniCheetahRunnerCfg.runner):
        run_name = ''
        experiment_name = 'mini_cheetah_ref'
        max_iterations = 1000  # number of policy updates
        algorithm_class_name = 'PPO'
        num_steps_per_env = 32

from gym.envs.base.legged_robot_config \
    import LeggedRobotCfg, LeggedRobotRunnerCfg

BASE_HEIGHT_REF = 0.33


class MiniCheetahCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 2**12
        num_actuators = 12
        episode_length_s = 10

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'

    class init_state(LeggedRobotCfg.init_state):
        default_joint_angles = {
            "haa": 0.0,
            "hfe": -0.785398,
            "kfe": 1.596976,
        }

        # * reset setup chooses how the initial conditions are chosen.
        # * "reset_to_basic" = a single position
        # * "reset_to_range" = uniformly random from a range defined below
        reset_mode = "reset_to_basic"

        # * default COM for basic initialization
        pos = [0.0, 0.0, 0.33]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # * initialization for random range setup
        dof_pos_range = {'haa': [-0.05, 0.05],
                         'hfe': [-0.85, -0.6],
                         'kfe': [-1.45, 1.72]}
        dof_vel_range = {'haa': [0., 0.],
                         'hfe': [0., 0.],
                         'kfe': [0., 0.]}
        root_pos_range = [[0., 0.],       # x
                          [0., 0.],       # y
                          [0.37, 0.4],    # z
                          [0., 0.],       # roll
                          [0., 0.],       # pitch
                          [0., 0.]]       # yaw
        root_vel_range = [[-0.05, 0.05],  # x
                          [0., 0.],       # y
                          [-0.05, 0.05],  # z
                          [0., 0.],       # roll
                          [0., 0.],       # pitch
                          [0., 0.]]       # yaw

    class control(LeggedRobotCfg.control):
        # * PD Drive parameters:
        stiffness = {'haa': 20., 'hfe': 20., 'kfe': 20.}
        damping = {'haa': 0.5, 'hfe': 0.5, 'kfe': 0.5}
        ctrl_frequency = 100
        desired_sim_frequency = 1000

    class commands:
        # * time before command are changed[s]
        resampling_time = 10.

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = 1.           # max [m/s]
            yaw_vel = 1              # max [rad/s]

    class push_robots:
        toggle = False
        interval_s = 1
        max_push_vel_xy = 0.5

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/" \
               + "mini_cheetah/urdf/mini_cheetah_simple.urdf"
        foot_name = "foot"
        penalize_contacts_on = []
        terminate_after_contacts_on = ["base", "thigh"]
        end_effector_names = ['foot']
        # * merge bodies connected by fixed joints.
        collapse_fixed_joints = False
        # * 1 to disable, 0 to enable...bitwise filter
        self_collisions = 1
        flip_visual_attachments = False
        disable_gravity = False
        # * set all torques set to 0
        disable_motors = False
        joint_damping = 0.1
        rotor_inertia = [0.002268, 0.002268, 0.005484]*4

    class reward_settings(LeggedRobotCfg.reward_settings):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 600.
        base_height_target = BASE_HEIGHT_REF
        tracking_sigma = 0.25

    class scaling(LeggedRobotCfg.scaling):
        base_ang_vel = 1.
        base_lin_vel = 1.
        commands = 1
        dof_vel = 100.  # ought to be roughly max expected speed.
        base_height = 1
        dof_pos = 1
        dof_pos_obs = dof_pos
        # * Action scales
        dof_pos_target = dof_pos
        # tau_ff = 4*[18, 18, 28]  # hip-abad, hip-pitch, knee
        clip_actions = 1000.


class MiniCheetahRunnerCfg(LeggedRobotRunnerCfg):
    seed = -1

    class policy(LeggedRobotRunnerCfg.policy):
        actor_hidden_dims = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = 'elu'

        actor_obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel"
            ]
        critic_obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel"
            ]

        actions = ["dof_pos_target"]
        add_noise = False

        class noise:
            dof_pos_obs = 0.005  # can be made very low
            dof_vel = 0.005
            base_ang_vel = 0.05
            projected_gravity = 0.02

        class reward(LeggedRobotRunnerCfg.policy.reward):
            class weights(LeggedRobotRunnerCfg.policy.reward.weights):
                tracking_lin_vel = 5.0
                tracking_ang_vel = 5.0
                lin_vel_z = 0.
                ang_vel_xy = 0.0
                orientation = 1.0
                torques = 5.e-7
                dof_vel = 1.
                base_height = 1.
                action_rate = 0.001
                action_rate2 = 0.0001
                stand_still = 0.
                dof_pos_limits = 0.
                feet_contact_forces = 0.
                dof_near_home = 1.

            class termination_weight:
                termination = 0.01

    class algorithm(LeggedRobotRunnerCfg.algorithm):
        # * training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 6
        # * mini batch size = num_envs*nsteps / nminibatches
        num_mini_batches = 6
        learning_rate = 1.e-4
        schedule = 'adaptive'  # can be adaptive or fixed
        discount_horizon = 1.  # [s]
        GAE_bootstrap_horizon = 1.  # [s]
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner(LeggedRobotRunnerCfg.runner):
        run_name = ''
        experiment_name = 'mini_cheetah'
        # * number of policy updates
        max_iterations = 1000
        algorithm_class_name = 'PPO'
        # * per iteration
        # * (n_steps in Rudin 2021 paper - batch_size = n_steps * n_robots)
        num_steps_per_env = 24

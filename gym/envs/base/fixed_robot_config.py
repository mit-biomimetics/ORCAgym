
from .base_config import BaseConfig


class FixedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_actuators = 1
        env_spacing = 4.  # not used with heightfields/trimeshes
        root_height = 2.
        episode_length_s = 4  # episode length in seconds

    class terrain:
        mesh_type = 'none'
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

    class init_state:
        # * reset_mode chooses how the initial conditions are chosen.
        # * "reset_to_basic" = a single position
        # * "reset_to_range" = uniformly random from a range defined below
        reset_mode = "reset_to_basic"

        # * target state when action = 0, also reset positions for basic mode
        default_joint_angles = {"joint_a": 0.,
                                "joint_b": 0.}

        # * initial conditiosn for reset_to_range
        dof_pos_range = {'joint_a': [-1., 1.],
                         'joint_b': [-1., 1.]}
        dof_vel_range = {'joint_a': [-1., 1.],
                         'joint_b': [-1., 1.]}

    class control:
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0}  # [N*m/rad]
        damping = {'joint_a': 0.5}     # [N*m*s/rad]

        actuated_joints_mask = []  # for each dof: 1 if actuated, 0 if passive

        ctrl_frequency = 100
        desired_sim_frequency = 100

    class asset:
        file = ""
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        disable_motors = False
        # * merge bodies connected by fixed joints.
        # * Specific fixed joints can be kept by adding
        # * " <... dont_collapse="true">
        collapse_fixed_joints = True
        fix_base_link = True  # fix the base of the robot
        # * see GymDofDriveModeFlags
        # * (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        default_dof_drive_mode = 3
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        # * replace collision cylinders with capsules,
        # * leads to faster/more stable simulation
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

    class reward_settings:
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        # * centage of urdf limits, values above this limit are penalized
        soft_dof_pos_limit = 1.
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.  # ! may want to turn this off

    class scaling:
        dof_pos = 1.
        dof_vel = 1.

        # Action scales
        tau_ff = 10
        dof_pos_target = 0.25

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 10.0
            # * 2**24 -> needed for 8000 envs and more
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            # * 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2


class FixedRobotCfgPPO(BaseConfig):
    seed = -1
    runner_class_name = 'OnPolicyRunner'

    class logging():
        enable_local_saving = True

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = 'elu'
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

        actor_obs = [
            "observation_a",
            "observation_b",
            "these_need_to_be_atributes_(states)_of_the_robot_env"]

        critic_obs = [
            "observation_x",
            "observation_y",
            "critic_obs_can_be_the_same_or_different_than_actor_obs"]

        actions = ["tau_ff"]
        disable_actions = False

        class noise:
            noise = 0.1  # implement as needed, also in your robot class

        class rewards:
            class weights:
                torques = 0.
                dof_vel = 0.
                collision = 0.
                action_rate = 0.
                action_rate2 = 0.
                dof_pos_limits = 1.

            class termination_weight:
                termination = 0.0

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
        num_steps_per_env = 24  # per iteration
        max_iterations = 500  # number of policy updates

        # * logging
        # * check for potential saves every this many iterations
        save_interval = 50
        run_name = ''
        experiment_name = 'fixed_robot'

        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        device = 'cpu'

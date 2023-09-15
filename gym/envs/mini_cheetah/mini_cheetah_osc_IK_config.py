from gym.envs.mini_cheetah.mini_cheetah_osc_config \
    import MiniCheetahOscCfg, MiniCheetahOscRunnerCfg

BASE_HEIGHT_REF = 0.33


class MiniCheetahOscIKCfg(MiniCheetahOscCfg):
    class env(MiniCheetahOscCfg.env):
        num_envs = 4096
        num_actuators = 12
        episode_length_s = 1.

    class terrain(MiniCheetahOscCfg.terrain):
        pass

    class init_state(MiniCheetahOscCfg.init_state):
        pass

    class control(MiniCheetahOscCfg.control):
        pass

    class osc(MiniCheetahOscCfg.osc):
        pass

    class commands:
        resampling_time = 4.  # time before command are changed[s]

        class ranges:
            lin_vel_x = [-1., 4.]  # min max [m/s]
            lin_vel_y = 1.  # max [m/s]
            yaw_vel = 3.14    # max [rad/s]

    class push_robots:
        toggle = False
        interval_s = 10
        max_push_vel_xy = 0.2

    class domain_rand(MiniCheetahOscCfg.domain_rand):
        pass

    class asset(MiniCheetahOscCfg.asset):
        pass

    class reward_settings(MiniCheetahOscCfg.reward_settings):
        pass

    class scaling(MiniCheetahOscCfg.scaling):
        ik_pos_target = 0.015


class MiniCheetahOscIKRunnerCfg(MiniCheetahOscRunnerCfg):
    seed = -1

    class policy(MiniCheetahOscRunnerCfg.policy):
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
                     "oscillator_obs"
                     #  "oscillators_vel",
                     #  "grf",
                     #  "osc_coupling",
                     #  "osc_offset"
                     ]

        critic_obs = ["base_height",
                      "base_lin_vel",
                      "base_ang_vel",
                      "projected_gravity",
                      "commands",
                      "dof_pos_obs",
                      "dof_vel",
                      "oscillator_obs",
                      "oscillators_vel"
                      ]

        actions = ["ik_pos_target"]

        class noise(MiniCheetahOscRunnerCfg.policy.noise):
            pass

        class reward(MiniCheetahOscRunnerCfg.policy.reward):
            class weights(MiniCheetahOscRunnerCfg.policy.reward.weights):
                pass

            class termination_weight:
                termination = 15./100.

    class algorithm(MiniCheetahOscRunnerCfg.algorithm):
        pass

    class runner(MiniCheetahOscRunnerCfg.runner):
        run_name = 'IK'
        experiment_name = 'oscIK'
        max_iterations = 1000  # number of policy updates
        algorithm_class_name = 'PPO'
        num_steps_per_env = 32

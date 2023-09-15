from gym.envs.mini_cheetah.mini_cheetah_osc_config \
    import MiniCheetahOscCfg, MiniCheetahOscRunnerCfg

BASE_HEIGHT_REF = 0.33


class MiniCheetahFloquetCfg(MiniCheetahOscCfg):
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

    class commands(MiniCheetahOscCfg.commands):
        pass
        # resampling_time = 4.  # time before command are changed[s]

        class ranges(MiniCheetahOscCfg.commands.ranges):
            pass
            # lin_vel_x = [-1., 4.]  # min max [m/s]
            # lin_vel_y = 1.  # max [m/s]
            # yaw_vel = 6    # max [rad/s]

    class push_robots:
        toggle = True
        interval_s = 10
        max_push_vel_xy = 0.2

    class domain_rand(MiniCheetahOscCfg.domain_rand):
        pass

    class asset(MiniCheetahOscCfg.asset):
        pass

    class reward_settings(MiniCheetahOscCfg.reward_settings):
        pass

    class scaling(MiniCheetahOscCfg.scaling):
        pass

class MiniCheetahFloquetRunnerCfg(MiniCheetahOscRunnerCfg):
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
                     "oscillator_obs",
                     "dof_pos_target"
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
                      "dof_pos_target"
                      ]

        actions = ["dof_pos_target"]

        class noise(MiniCheetahOscRunnerCfg.policy.noise):
            pass

        class reward(MiniCheetahOscRunnerCfg.policy.reward):
            class weights(MiniCheetahOscRunnerCfg.policy.reward.weights):
                floquet = 0.
                locked_frequency = 0.

            class termination_weight:
                pass
                # termination = 15./100.

    class algorithm(MiniCheetahOscRunnerCfg.algorithm):
        pass

    class runner(MiniCheetahOscRunnerCfg.runner):
        run_name = 'floquet'
        experiment_name = 'osc'
        max_iterations = 3000  # number of policy updates
        algorithm_class_name = 'PPO'
        num_steps_per_env = 32

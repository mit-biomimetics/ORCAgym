def adjust_settings(toggle, env_cfg, train_cfg):
    # * settings
    if train_cfg.runner.experiment_name =='':
        train_cfg.runner.experiment_name = "ORC_" + toggle
    else:
        train_cfg.runner.experiment_name = "ORC_" + toggle + '_' \
                                            +train_cfg.runner.experiment_name
    train_cfg.runner.run_name = "ORC_"+toggle
    # task_registry.set_log_dir_name(train_cfg)
    toggle = [x for x in toggle]
    if toggle[0] == '0':
        # * No osc observation
        train_cfg.policy.actor_obs.remove('oscillator_obs')
        train_cfg.policy.critic_obs.remove('oscillator_obs')
        train_cfg.policy.critic_obs.remove('oscillators_vel')
    if toggle[1] == '0':
        # * non reward
        train_cfg.policy.reward.weights.swing_grf = 0.
        train_cfg.policy.reward.weights.stance_grf = 0.
    if toggle[2] == '0':
        # * no coupling
        env_cfg.osc.coupling = 0.
        env_cfg.osc.coupling_range = [0., 0.]
        coupling_stop = 0.
        coupling_step = 0.
        coupling_slope = 0.
        env_cfg.osc.coupling_max = 0.

    return env_cfg, train_cfg
from gym.envs import __init__
from gym.utils import get_args, task_registry
from gym.utils.logging_and_saving \
    import local_code_save_helper, wandb_singleton


def setup():
    args = get_args()
    wandb_helper = wandb_singleton.WandbSingleton()

    # * prepare environment
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    task_registry.make_gym_and_sim()
    wandb_helper.setup_wandb(env_cfg=env_cfg, train_cfg=train_cfg, args=args)
    env = task_registry.make_env(name=args.task, env_cfg=env_cfg)
    # * then make env
    policy_runner = task_registry.make_alg_runner(env, train_cfg)

    local_code_save_helper.log_and_save(
        env, env_cfg, train_cfg, policy_runner)
    wandb_helper.attach_runner(policy_runner=policy_runner)

    return train_cfg, policy_runner


def train(train_cfg, policy_runner):
    wandb_helper = wandb_singleton.WandbSingleton()

    policy_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True)

    wandb_helper.close_wandb()


if __name__ == '__main__':
    train_cfg, policy_runner = setup()
    train(train_cfg=train_cfg, policy_runner=policy_runner)

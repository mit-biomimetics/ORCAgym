from gym.envs import __init__
from gym.utils import get_args, task_registry
from gym.utils.logging_and_saving \
    import local_code_save_helper, wandb_singleton
from ORC import adjust_settings

from torch.multiprocessing import Process
from torch.multiprocessing import set_start_method

def setup(toggle):
    args = get_args()
    wandb_helper = wandb_singleton.WandbSingleton()

    # * prepare environment
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    env_cfg, train_cfg = adjust_settings(toggle=toggle,
                                         env_cfg=env_cfg,
                                         train_cfg=train_cfg)
    task_registry.set_log_dir_name(train_cfg)

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

def worker(toggle):
    train_cfg, policy_runner = setup(toggle)
    train(train_cfg=train_cfg, policy_runner=policy_runner)

if __name__ == '__main__':
    all_toggles = ['000', '010', '011', '100', '101', '110', '111']
    set_start_method('spawn')

    processes = []
    for toggle in all_toggles:
        p = Process(target=worker, args=(toggle,))
        p.start()
        p.join()  # Wait for the process to finish

        # Free up any resources if needed
        p.terminate()
        p.close()

    print("All learning runs are completed.")

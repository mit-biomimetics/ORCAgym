import os
import isaacgym
import torch

from gym.envs import __init__
from gym.utils.logging_and_saving import local_code_save_helper
from gym.utils import task_registry

from multiprocessing import Process, Manager


def _run_integration_test(return_queue, args):
    # * do initial setup
    env_cfg, train_cfg = task_registry.create_cfgs(args)

    task_registry.make_gym_and_sim()
    env = task_registry.make_env(name=args.task, env_cfg=env_cfg)
    policy_runner = task_registry.make_alg_runner(env=env, train_cfg=train_cfg)

    local_code_save_helper.log_and_save(env, env_cfg, train_cfg, policy_runner)

    # * get the default test values before learning
    default_actions = policy_runner.env.get_states(train_cfg.policy.actions)

    # * take a full iteration step
    policy_runner.learn(num_learning_iterations=args.max_iterations)

    # * get the test values after learning
    actions = policy_runner.env.get_states(train_cfg.policy.actions)

    # * return the values to the parent for assertion
    return_queue.put((actions.cpu().clone(), default_actions.cpu().clone()))
    return_queue.put(policy_runner.it)
    return_queue.put(policy_runner.log_dir)


class TestDefaultIntegration():
    def test_default_integration_settings(self, args):
        # * simulation settings for the test
        args.task = 'mini_cheetah'
        args.max_iterations = 1
        args.save_interval = 1
        args.num_envs = 16
        args.headless = True
        args.disable_wandb = True

        # * create a queue to return the values for assertion
        manager = Manager()
        return_queue = manager.Queue()

        # * spin up a child process to run the simulation iteration
        test_proc = Process(
            target=_run_integration_test, args=(return_queue, args))
        test_proc.start()
        test_proc.join()

        # * extract the values to test from the child's return queue
        actions, default_actions = return_queue.get()

        it = return_queue.get()
        log_dir = return_queue.get()

        model_0_path = os.path.join(log_dir, 'model_0.pt')
        model_1_path = os.path.join(log_dir, 'model_1.pt')

        # * assert the returned values are as expected
        assert torch.equal(actions, default_actions) is False, \
            'Actions were not updated from default'
        assert it == 1, 'Iteration update incorrect'
        assert os.path.exists(model_0_path), 'model_0.pt was not saved'
        assert os.path.exists(model_1_path), 'model_1.pt was not saved'

        # * kill the child process now that we're done with it
        test_proc.kill()

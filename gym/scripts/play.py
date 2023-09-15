import os

from gym.envs import __init__
from gym import LEGGED_GYM_ROOT_DIR
from gym.utils import get_args, task_registry
from gym.utils import KeyboardInterface, GamepadInterface

# torch needs to be imported after isaacgym imports in local source
import torch


def setup(args):
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False
    env_cfg.commands.resampling_time = 9999
    env_cfg.env.episode_length_s = 9999
    env_cfg.init_state.timeout_reset_ratio = 1.
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.terrain.mesh_type = "plane"

    task_registry.make_gym_and_sim()
    env = task_registry.make_env(args.task, env_cfg)
    env.cfg.init_state.reset_mode = "reset_to_basic"
    train_cfg.runner.resume = True
    runner = task_registry.make_alg_runner(env, train_cfg)

    # * switch to evaluation mode (dropout for example)
    runner.switch_to_eval()
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                            train_cfg.runner.experiment_name, 'exported')
        runner.export(path)
    return env, runner, train_cfg


def play(env, runner, train_cfg):

    # * set up interface: GamepadInterface(env) or KeyboardInterface(env)
    COMMANDS_INTERFACE = hasattr(env, "commands")
    if COMMANDS_INTERFACE:
        # interface = GamepadInterface(env)
        interface = KeyboardInterface(env)
    for i in range(10*int(env.max_episode_length)):
        if COMMANDS_INTERFACE:
            interface.update(env)
        runner.set_actions(runner.get_inference_actions())
        env.step()


if __name__ == '__main__':
    EXPORT_POLICY = True
    args = get_args()
    with torch.no_grad():
        env, runner, train_cfg = setup(args)
        play(env, runner, train_cfg)

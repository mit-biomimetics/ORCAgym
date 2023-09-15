import os

from gym.envs import __init__
from gym import LEGGED_GYM_ROOT_DIR
from gym.utils import get_args, task_registry
from gym.utils import KeyboardInterface, GamepadInterface
from ORC import adjust_settings
# torch needs to be imported after isaacgym imports in local source
import torch


def setup(args):
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    env_cfg.env.num_envs = 50
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False
    env_cfg.commands.resampling_time = 9999
    env_cfg.env.episode_length_s = 9999
    env_cfg.init_state.timeout_reset_ratio = 1.
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.osc.init_to = 'random'
    env_cfg.osc.process_noise = 0.
    env_cfg.osc.omega_var = 0.
    env_cfg.osc.coupling_var = 0.
    env_cfg.commands.ranges.lin_vel_x = [0., 0.]
    env_cfg.commands.ranges.lin_vel_y = 0.
    env_cfg.commands.ranges.yaw_vel = 0.
    env_cfg.commands.var = 0.

    train_cfg.policy.noise.scale = 1.0

    env_cfg, train_cfg = adjust_settings(toggle=args.ORC_toggle,
                                         env_cfg=env_cfg,
                                         train_cfg=train_cfg)
    env_cfg.init_state.reset_mode = "reset_to_basic"
    task_registry.set_log_dir_name(train_cfg)

    task_registry.make_gym_and_sim()
    env = task_registry.make_env(args.task, env_cfg)
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
    RECORD_FRAMES = False
    # * set up interface: GamepadInterface(env) or KeyboardInterface(env)
    COMMANDS_INTERFACE = hasattr(env, "commands")
    if COMMANDS_INTERFACE:
        # interface = GamepadInterface(env)
        interface = KeyboardInterface(env)
    img_idx = 0

    for i in range(10*int(env.max_episode_length)):

        if RECORD_FRAMES:
            if i % 5:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR,
                                        'gym', 'scripts', 'gifs',
                                        train_cfg.runner.experiment_name,
                                        f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1

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

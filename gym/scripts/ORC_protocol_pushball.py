import os

from gym.envs import __init__
from gym import LEGGED_GYM_ROOT_DIR
from gym.utils import get_args, task_registry
from gym.utils import KeyboardInterface, GamepadInterface
from ORC import adjust_settings
# torch needs to be imported after isaacgym imports in local source
import torch
from torch.multiprocessing import Process
from torch.multiprocessing import set_start_method
import numpy as np
import glob
import imageio
from datetime import datetime

RECORD_VID = False

def get_run_names(experiment_name):
    experiment_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                                   experiment_name)
    return [folder_name for folder_name in os.listdir(experiment_path)
            if os.path.isdir(os.path.join(experiment_path, folder_name))]

def create_logging_dict(runner, test_total_timesteps):
    # states_to_log = ['dof_pos_target',
    #                  'dof_pos_obs',
    #                  'dof_vel',
    #                  'torques',
    #                  'commands',
    #                  'base_lin_vel',
    #                  'base_ang_vel',
    #                  'oscillators',
    #                  'grf',
    #                  'base_height']
    states_to_log = [
                     'commands',
                     'base_lin_vel',
                     'base_ang_vel',
                     'oscillators',
                     'base_height']
    states_to_log_dict = {}

    for state in states_to_log:
        array_dim = runner.get_obs_size([state, ])
        states_to_log_dict[state] = torch.zeros((runner.env.num_envs,
                                                 test_total_timesteps,
                                                 array_dim),
                                                device=runner.env.device)
    return states_to_log, states_to_log_dict

def setup(toggle, run_name):
    args = get_args()
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    env_cfg.env.num_envs = 1800
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False
    env_cfg.commands.resampling_time = 999999
    env_cfg.env.episode_length_s = 100.
    env_cfg.init_state.timeout_reset_ratio = 1.
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.osc.init_to = 'random'
    env_cfg.osc.init_w_offset = False
    env_cfg.osc.process_noise = 0.
    env_cfg.osc.omega_var = 0.
    env_cfg.osc.coupling_var = 0.
    env_cfg.commands.ranges.lin_vel_x = [0., 0.]
    env_cfg.commands.ranges.lin_vel_y = 0.
    env_cfg.commands.ranges.yaw_vel = 0.
    env_cfg.commands.var = 0.
    env_cfg.env.env_spacing = 0.0
    train_cfg.policy.noise.scale = 0.0
    train_cfg.runner.run_name = run_name
    train_cfg.runner.load_run = run_name
    env_cfg, train_cfg = adjust_settings(toggle=toggle,
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
    # ... [rest of your setup code]
    return env, runner, train_cfg


def play(env, runner, train_cfg, png_folder):
    protocol_name = 'pushball_run'+train_cfg.runner.run_name

    # * set up logging
    # log_file_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'gym', 'scripts',
    #                              "osc_data", train_cfg.runner.run_name,
    #                              protocol_name+".npz")
    # if not os.path.exists(os.path.dirname(log_file_path)):
    #     os.makedirs(os.path.dirname(log_file_path))

    push_interval = 0.5
    stagger_interval = 0.01
    num_staggers = int(push_interval/stagger_interval)
    stagger_timesteps = int(stagger_interval/env.cfg.control.ctrl_dt)
    n_directions = 36
    n_trials = num_staggers*n_directions  # ! this must match num_envs
    # print(f"number of trials: {n_trials}")
    # print(f"number of loaded environments: {env.num_envs}")
    assert n_trials == env.num_envs, f"number of trials: {n_trials}, number of loaded environments: {env.num_envs}"
    env.commands[:, 0] = 3.
    push_magnitude = 3.0
    angles = torch.linspace(0, 2*np.pi, n_directions, device=env.device )
    push_ball = torch.stack((torch.cos(angles),
                             torch.sin(angles),
                             torch.zeros_like(angles)), dim=1)*push_magnitude

    kdx = 0  # trial counter
    img_idx = 0

    entrainment_time = 5.

    test_start_time = 1.0
    test_total_time = 9.0 + test_start_time
    test_total_timesteps = int(test_total_time/env.cfg.control.ctrl_dt)

    states_to_log, states_to_log_dict = create_logging_dict(runner,
                                                        test_total_timesteps)

    for t in range(int(entrainment_time/env.cfg.control.ctrl_dt)):
        # * sim step
        runner.set_actions(runner.get_inference_actions())
        env.step()

    for t in range(int(test_total_time/env.cfg.control.ctrl_dt)):

        # * record pngs
        if RECORD_VID and t % 5:
            filename = os.path.join(png_folder,
                                    f"{img_idx}.png")
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            img_idx += 1

        # * do your thing here
        if t == int(test_start_time/env.cfg.control.ctrl_dt) + stagger_timesteps*kdx:
            env_ids = torch.arange(kdx*n_directions,
                                   (kdx+1)*n_directions,
                                   dtype=torch.int64,
                                   device=env.device)
            env.perturb_base_velocity(push_ball, env_ids)
            kdx += 1
        if kdx == num_staggers:  # turn off
            kdx = 0
        # * sim step
        runner.set_actions(runner.get_inference_actions())
        env.step()

        # * log
        for state in states_to_log:
            states_to_log_dict[state][:, t, :] = getattr(env, state)

    # * save data
    return states_to_log_dict
    # np.savez_compressed(log_file_path, **states_to_log_dict)
    # print("saved to ", log_file_path)


def save_gif(png_folder, gif_folder, frame_rate):
    png_files = sorted(glob.glob(f"{png_folder}/*.png"),
                       key=os.path.getmtime)
    images = [imageio.imread(f) for f in png_files]
    gif_path = os.path.join(gif_folder, 'output.gif')
    imageio.mimsave(gif_path, images, fps=frame_rate)


def worker(toggle, run_name):
    with torch.no_grad():
        env, runner, train_cfg = setup(toggle, run_name)
        # ... [rest of your worker code]

        # Adjust log_file_path to include run_name
        log_file_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'gym', 'scripts',
                                     "FS_data", train_cfg.runner.run_name,
                                     run_name+"_data.npz")

        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        png_folder = os.path.join(LEGGED_GYM_ROOT_DIR, 'temp', run_name)
        os.makedirs(png_folder, exist_ok=True)

        states_to_log_dict = play(env, runner=runner, train_cfg=train_cfg,
                                  png_folder=png_folder)

        states_to_log_dict_cpu = {k: v.detach().cpu().numpy()
                                  for k, v in states_to_log_dict.items()}

        np.savez_compressed(log_file_path, **states_to_log_dict_cpu)

        gif_folder = os.path.join(LEGGED_GYM_ROOT_DIR, 'gym', 'scripts',
                                  "osc_data", train_cfg.runner.run_name,
                                  run_name)
        if RECORD_VID:
            save_gif(png_folder, gif_folder, frame_rate=20)


if __name__ == '__main__':
    set_start_method('spawn')
    # all_toggles = ['111']
    all_toggles = ['000', '111', '110', '100', '101', '001', '010', '011']
    experiment_name = "ORC_" + all_toggles[0] + "_FullSend"
    run_names = get_run_names(experiment_name)
    worker(all_toggles[0], run_names[0])


    for toggle in all_toggles:
        experiment_name = "ORC_" + toggle+ "_FullSend"
        run_names = get_run_names(experiment_name)

        for run_name in run_names:
            p = Process(target=worker, args=(toggle, run_name))
            p.start()
            p.join()

            # Free up any resources if needed
            p.terminate()
            p.close()

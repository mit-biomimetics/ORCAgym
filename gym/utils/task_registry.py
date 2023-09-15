# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
from isaacgym import gymapi
from isaacgym import gymutil
from datetime import datetime
from typing import Tuple

from learning.env import VecEnv
from learning.runners import OnPolicyRunner
from learning.utils import set_discount_from_horizon

from gym import LEGGED_GYM_ROOT_DIR
from .helpers import \
    update_cfg_from_args, class_to_dict, get_load_path, set_seed
from gym.envs.base.legged_robot_config import (
    LeggedRobotCfg, LeggedRobotRunnerCfg)
from gym.envs.base.base_config import BaseConfig
from gym.envs.base.sim_config import SimCfg


class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
        self.sim_cfg = class_to_dict(SimCfg)
        self.sim = {}

    def register(self, name: str, task_class: VecEnv,
                 env_cfg: BaseConfig, train_cfg: LeggedRobotRunnerCfg):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]

    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotRunnerCfg]:
        env_cfg = self.env_cfgs[name]
        train_cfg = self.train_cfgs[name]
        # * copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    def create_cfgs(self, args):
        env_cfg, train_cfg = self.get_cfgs(name=args.task)
        self.update_and_parse_cfgs(env_cfg, train_cfg, args)
        self.set_log_dir_name(train_cfg)
        return env_cfg, train_cfg

    def set_log_dir_name(self, train_cfg, log_root="default"):
        if log_root == "default":
            log_root = os.path.join(
                LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(
                log_root, datetime.now().strftime('%b%d_%H-%M-%S')
                + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        train_cfg.log_dir = log_dir

    def update_and_parse_cfgs(self, env_cfg, train_cfg, args):
        update_cfg_from_args(env_cfg, train_cfg, args)
        self.convert_frequencies_to_params(env_cfg, train_cfg)
        self.update_sim_cfg(args)

    def convert_frequencies_to_params(self, env_cfg, train_cfg):
        self.set_control_and_sim_dt(env_cfg, train_cfg)
        self.set_discount_rates(train_cfg, env_cfg.control.ctrl_dt)

    def set_control_and_sim_dt(self, env_cfg, train_cfg):
        env_cfg.control.decimation = int(env_cfg.control.desired_sim_frequency
                                         / env_cfg.control.ctrl_frequency)
        env_cfg.control.ctrl_dt = 1.0 / env_cfg.control.ctrl_frequency
        env_cfg.sim_dt = env_cfg.control.ctrl_dt / env_cfg.control.decimation
        self.sim_cfg["dt"] = env_cfg.sim_dt
        if env_cfg.sim_dt != 1.0/env_cfg.control.desired_sim_frequency:
            print(f'****** Simulation dt adjusted from '
                  f'{1.0/env_cfg.control.desired_sim_frequency}'
                  f' to {env_cfg.sim_dt}.')

    def set_discount_rates(self, train_cfg, dt):
        if hasattr(train_cfg.algorithm, 'discount_horizon'):
            hrzn = train_cfg.algorithm.discount_horizon
            train_cfg.algorithm.gamma = set_discount_from_horizon(dt, hrzn)

        if hasattr(train_cfg.algorithm, 'GAE_bootstrap_horizon'):
            hrzn = train_cfg.algorithm.GAE_bootstrap_horizon
            train_cfg.algorithm.lam = set_discount_from_horizon(dt, hrzn)

    def update_sim_cfg(self, args):
        self.sim["sim_device"] = args.sim_device
        self.sim["sim_device_id"] = args.sim_device_id
        self.sim["graphics_device_id"] = args.graphics_device_id
        self.sim["physics_engine"] = args.physics_engine
        self.sim["headless"] = args.headless
        if self.sim["headless"]:
            self.sim["graphics_device_id"] = -1
        self.sim["params"] = gymapi.SimParams()
        self.sim["params"].physx.use_gpu = args.use_gpu
        self.sim["params"].physx.num_subscenes = args.subscenes
        self.sim["params"].use_gpu_pipeline = args.use_gpu_pipeline
        gymutil.parse_sim_config(self.sim_cfg, self.sim["params"])

    def make_gym_and_sim(self):
        self.make_gym()
        self.make_sim()

    def make_gym(self):
        self._gym = gymapi.acquire_gym()

    def make_sim(self):
        self._sim = self._gym.create_sim(
            self.sim["sim_device_id"],
            self.sim["graphics_device_id"],
            self.sim["physics_engine"],
            self.sim["params"])

    def make_env(self, name, env_cfg) -> VecEnv:
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        set_seed(env_cfg.seed)
        env = task_class(gym=self._gym, sim=self._sim, cfg=env_cfg,
                         sim_params=self.sim["params"],
                         sim_device=self.sim["sim_device"],
                         headless=self.sim["headless"])
        return env

    def make_alg_runner(self, env, train_cfg):
        train_cfg_dict = class_to_dict(train_cfg)
        runner = OnPolicyRunner(env, train_cfg_dict, train_cfg.runner.device)

        # * save resume path before creating a new log_dir
        if train_cfg.runner.resume:
            resume_path = get_load_path(name=train_cfg.runner.experiment_name,
                                        load_run=train_cfg.runner.load_run,
                                        checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        return runner


# make global task registry
task_registry = TaskRegistry()

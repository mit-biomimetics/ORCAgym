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

import time
import os
import wandb
import torch
from isaacgym.torch_utils import torch_rand_float

from learning.algorithms import PPO
from learning.modules import ActorCritic
from learning.env import VecEnv
from learning.utils import remove_zero_weighted_rewards
from learning.utils import Logger


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 device='cpu'):

        self.device = device
        self.env = env
        self.parse_train_cfg(train_cfg)

        num_actor_obs = self.get_obs_size(self.policy_cfg["actor_obs"])
        num_critic_obs = self.get_obs_size(self.policy_cfg["critic_obs"])
        num_actions = self.get_action_size(self.policy_cfg["actions"])
        actor_critic = ActorCritic(num_actor_obs,
                                   num_critic_obs,
                                   num_actions,
                                   **self.policy_cfg).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: PPO = alg_class(actor_critic,
                                  device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.tot_timesteps = 0
        self.tot_time = 0
        self.it = 0

        # * init storage and model
        self.init_storage()

        # * Log
        self.log_dir = train_cfg["log_dir"]
        self.SE_path = os.path.join(self.log_dir, 'SE')  # log_dir for SE
        self.logger = Logger(self.log_dir, self.env.max_episode_length_s,
                             self.device)

        reward_keys_to_log = \
            list(self.policy_cfg["reward"]["weights"].keys()) \
            + list(self.policy_cfg["reward"]["termination_weight"].keys())

        reward_keys_to_log += ["Total_reward"]

        # ! gait tracking
        self.gait_weights = {'trot': 1.,
                             'pace': 1.,
                             'pronk': 1.,
                             'bound': 1,
                             'asymettric': 1.}
        reward_keys_to_log += list(self.gait_weights.keys())
        reward_keys_to_log += ['Gait_score']
        self.logger.initialize_buffers(self.env.num_envs, reward_keys_to_log)

    def parse_train_cfg(self, train_cfg):
        self.cfg = train_cfg['runner']
        self.alg_cfg = train_cfg['algorithm']

        remove_zero_weighted_rewards(train_cfg['policy']['reward']['weights'])
        self.policy_cfg = train_cfg['policy']

    def init_storage(self):
        num_actor_obs = self.get_obs_size(self.policy_cfg["actor_obs"])
        num_critic_obs = self.get_obs_size(self.policy_cfg["critic_obs"])
        num_actions = self.get_action_size(self.policy_cfg["actions"])
        self.alg.init_storage(self.env.num_envs,
                              self.num_steps_per_env,
                              actor_obs_shape=[num_actor_obs],
                              critic_obs_shape=[num_critic_obs],
                              action_shape=[num_actions])

    def attach_to_wandb(self, wandb, log_freq=100, log_graph=True):
        wandb.watch((self.alg.actor_critic.actor,
                    self.alg.actor_critic.critic),
                    log_freq=log_freq,
                    log_graph=log_graph)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length))

        actor_obs = self.get_obs(self.policy_cfg["actor_obs"])
        critic_obs = self.get_obs(self.policy_cfg["critic_obs"])
        self.alg.actor_critic.train()
        self.num_learning_iterations = num_learning_iterations
        self.tot_iter = self.it + num_learning_iterations

        self.save()

        reward_weights = self.policy_cfg['reward']['weights']
        termination_weight = self.policy_cfg['reward']['termination_weight']
        rewards = 0.*self.get_rewards(reward_weights)

        # * simulate 2 seconds to let the robots fall
        for i in range(int(2/self.env.dt)):
            self.env.step()

        for self.it in range(self.it+1, self.tot_iter+1):
            start = time.time()
            # * Rollout
            with torch.inference_mode():

                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(actor_obs, critic_obs)
                    self.set_actions(actions)

                    self.env.step()

                    actor_obs = self.get_noisy_obs(
                        self.policy_cfg['actor_obs'],
                        self.policy_cfg['noise'])
                    critic_obs = self.get_obs(self.policy_cfg['critic_obs'])
                    # * get time_outs
                    timed_out = self.get_timed_out()
                    terminated = self.get_terminated()
                    dones = timed_out | terminated

                    rewards += self.get_and_log_rewards(reward_weights,
                                                        modifier=self.env.dt,
                                                        mask=~terminated)
                    rewards += self.get_and_log_rewards(termination_weight,
                                                        mask=terminated)
                    self.logger.log_current_reward('Total_reward', rewards)

                    # * log for gaits
                    gait_score = self.get_and_log_rewards(self.gait_weights,
                                                          modifier=self.env.dt,
                                                          mask=~terminated)
                    self.logger.log_current_reward('Gait_score', gait_score)
                    self.alg.process_env_step(rewards, dones, timed_out)
                    self.logger.update_episode_buffer(dones)
                    rewards *= 0.

                stop = time.time()
                self.collection_time = stop - start
                # * Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            self.mean_value_loss, self.mean_surrogate_loss = self.alg.update()
            stop = time.time()
            self.learn_time = stop - start
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            self.tot_time += self.collection_time + self.learn_time

            self.log()

            if self.it % self.save_interval == 0:
                self.save()

        # * save
        self.save()

    def get_noise(self, obs_list, noise_dict):
        noise_vec = torch.zeros(self.get_obs_size(obs_list),
                                device=self.device)
        obs_index = 0
        for obs in obs_list:
            obs_size = self.get_obs_size([obs])
            if obs in noise_dict.keys():
                noise_tensor = torch.ones(obs_size).to(self.device) \
                               * torch.tensor(noise_dict[obs]).to(self.device)
                if obs in self.env.scales.keys():
                    noise_tensor /= self.env.scales[obs]
                noise_vec[obs_index:obs_index+obs_size] = noise_tensor
            obs_index += obs_size
        return torch_rand_float(-1., 1., (self.env.num_envs, len(noise_vec)),
                                self.device) * noise_vec * noise_dict["scale"]

    def get_noisy_obs(self, obs_list, noise_dict):
        observation = self.get_obs(obs_list)
        return observation + self.get_noise(obs_list, noise_dict)

    def get_obs(self, obs_list):
        observation = self.env.get_states(obs_list).to(self.device)
        return observation

    def set_actions(self, actions):
        if self.policy_cfg['disable_actions']:
            return
        if hasattr(self.env.cfg.scaling, "clip_actions"):
            actions = torch.clip(actions,
                                 -self.env.cfg.scaling.clip_actions,
                                 self.env.cfg.scaling.clip_actions)
        self.env.set_states(self.policy_cfg["actions"], actions)

    def get_timed_out(self):
        return self.env.get_states(['timed_out']).to(self.device)

    def get_terminated(self):
        return self.env.get_states(['terminated']).to(self.device)

    def get_obs_size(self, obs_list):
        # todo make unit-test to assert len(shape)==1 always
        return self.get_obs(obs_list)[0].shape[0]

    def get_action_size(self, action_list):
        return self.env.get_states(action_list)[0].shape[0]

    def get_and_log_rewards(self, reward_weights, modifier=1,
                            mask=None):
        '''
        Computes each reward on the fly, sends them to logging, and returns the
        total reward.
        reward_weights: dict with reward name, and weighting
        modifier: an additional weighting applied to all rewards
        mask: a boolean tensor of shape (num_envs), to toggle which rewards are
              computed
        '''

        if mask is None:
            mask = 1.0
        total_rewards = torch.zeros(self.env.num_envs,
                                    device=self.device, dtype=torch.float)
        for name, weight in reward_weights.items():
            reward = mask * self.get_rewards({name: weight}, modifier)
            total_rewards += reward
            self.logger.log_current_reward(name, reward)
        return total_rewards

    def get_rewards(self, reward_weights, modifier=1):
        return modifier*self.env.compute_reward(reward_weights).to(self.device)

    def log(self):
        fps = int(self.num_steps_per_env * self.env.num_envs
                  / (self.collection_time+self.learn_time))
        mean_noise_std = self.alg.actor_critic.std.mean().item()
        self.logger.add_log(self.logger.mean_rewards)
        self.logger.add_log({
            'Loss/value_function': self.mean_value_loss,
            'Loss/surrogate': self.mean_surrogate_loss,
            'Loss/learning_rate': self.alg.learning_rate,
            'Policy/mean_noise_std': mean_noise_std,
            'Perf/total_fps': fps,
            'Perf/collection_time': self.collection_time,
            'Perf/learning_time': self.learn_time,
            'Train/mean_reward': self.logger.total_mean_reward,
            'Train/mean_episode_length': self.logger.mean_episode_length,
            'Train/total_timesteps': self.tot_timesteps,
            'Train/iteration_time': self.collection_time+self.learn_time,
            'Train/time': self.tot_time,
            })
        self.logger.update_iterations(self.it, self.tot_iter,
                                      self.num_learning_iterations)

        # TODO: iterate through the config for any extra things
        # TODO: you might want to log

        if wandb.run is not None:
            self.logger.log_to_wandb()
        self.logger.print_to_terminal()

    def get_infos(self):
        return self.env.extras

    def save(self):
        path = os.path.join(self.log_dir, 'model_{}.pt'.format(self.it))
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.it},
                   path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(
                loaded_dict['optimizer_state_dict'])
        self.it = loaded_dict['iter']

    def switch_to_eval(self):
        self.alg.actor_critic.eval()

    def get_inference_actions(self):
        obs = self.get_noisy_obs(self.policy_cfg["actor_obs"],
                                 self.policy_cfg['noise'])
        return self.alg.actor_critic.actor.act_inference(obs)

    def export(self, path):
        self.alg.actor_critic.export_policy(path)

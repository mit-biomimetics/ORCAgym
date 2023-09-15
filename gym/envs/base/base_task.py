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

import sys
from isaacgym import gymapi
from isaacgym import gymutil
import torch


# * Base class for RL tasks
class BaseTask():
    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        self.gym = gym
        self.sim = sim
        self.sim_params = sim_params
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = \
            gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # * env device is GPU only if sim is on GPU and use_gpu_pipeline=True,
        # * otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # * graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id

        self.num_envs = cfg.env.num_envs
        self.num_actuators = cfg.env.num_actuators

        # * optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.to_be_reset = torch.ones(self.num_envs,
                                      device=self.device,
                                      dtype=torch.bool)
        self.terminated = torch.ones(self.num_envs,
                                     device=self.device,
                                     dtype=torch.bool)
        self.episode_length_buf = torch.zeros(self.num_envs,
                                              device=self.device,
                                              dtype=torch.long)
        self.timed_out = torch.zeros(self.num_envs,
                                     device=self.device,
                                     dtype=torch.bool)

        self.extras = {}

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # * if running with a viewer, set up keyboard shortcuts and camera
        if self.headless is False:
            # * subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

    def get_states(self, obs_list):
        return torch.cat([self.get_state(obs) for obs in obs_list], dim=-1)

    def get_state(self, name):
        if name in self.scales.keys():
            return getattr(self, name)/self.scales[name]
        else:
            return getattr(self, name)

    def set_states(self, state_list, values):
        idx = 0
        for state in state_list:
            state_dim = getattr(self, state).shape[1]
            self.set_state(state, values[:, idx:idx+state_dim])
            idx += state_dim
        assert (idx == values.shape[1]), "Actions don't equal tensor shapes"

    def set_state(self, name, value):
        try:
            if name in self.scales.keys():
                setattr(self, name, value*self.scales[name])
            else:
                setattr(self, name, value)
        except AttributeError:
            print("Value for " + name + " does not match tensor shape")

    def _reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self.step()

    def _reset_buffers(self):
        self.to_be_reset[:] = False
        self.terminated[:] = False
        self.timed_out[:] = False

    def compute_reward(self, reward_weights):
        ''' Compute and return a torch tensor of rewards
        reward_weights: dict with keys matching reward names, and values
            matching weights
        '''
        reward = torch.zeros(self.num_envs,
                             device=self.device, dtype=torch.float)
        for name, weight in reward_weights.items():
            reward += weight*self._eval_reward(name)
        return reward

    def _eval_reward(self, name):
        return eval('self._reward_'+name+'()')

    def _check_terminations_and_timeouts(self):
        """ Check if environments need to be reset
        """
        contact_forces = \
            self.contact_forces[:, self.termination_contact_indices, :]
        self.terminated = \
            torch.any(torch.norm(contact_forces, dim=-1) > 1., dim=1)
        self.timed_out = self.episode_length_buf > self.max_episode_length
        self.to_be_reset = self.timed_out | self.terminated

    def step(self, actions):
        raise NotImplementedError

    def _render(self, sync_frame_time=True):
        if self.viewer:
            # * check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # * check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # * fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # * step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
                # self.gym.draw_viewer(self.viewer, self.sim, True)

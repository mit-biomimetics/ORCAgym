import torch

from .base_storage import BaseStorage


class SERolloutStorage(BaseStorage):
    """ Store episodic data for supervised learning of the state-estimator.
    """

    class Transition:
        def __init__(self):
            self.observations = None
            self.SE_targets = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, actor_obs_shape,
                 se_shape, device='cpu'):

        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.fill_count = 0

        self.actor_obs_shape = actor_obs_shape    # raw states for actor
        self.se_shape = se_shape                  # SE prediction states

        self.observations = torch.zeros(num_transitions_per_env, num_envs,
                                        *actor_obs_shape, device=self.device)
        self.SE_targets = torch.zeros(num_transitions_per_env, num_envs,
                                      *se_shape, device=self.device)

    def add_transitions(self, transition: Transition):
        if self.fill_count >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.fill_count].copy_(transition.observations)
        self.SE_targets[self.fill_count].copy_(transition.SE_targets)
        self.fill_count += 1

    def clear(self):
        self.fill_count = 0

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """ Generate mini batch for learning
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size,
                                 requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        SE_targets = self.SE_targets.flatten(0, 1)
        # if self.privileged_observations is not None:
        #     critic_observations = self.privileged_observations.flatten(0, 1)
        # else:
        #     critic_observations = observations

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                # critic_observations_batch = critic_observations[batch_idx]
                SE_target_batch = SE_targets[batch_idx]
                yield obs_batch, SE_target_batch

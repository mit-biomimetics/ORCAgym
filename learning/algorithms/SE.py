import torch.nn as nn
import torch.optim as optim

from learning.modules import StateEstimatorNN
from learning.storage import SERolloutStorage


class StateEstimator:
    """ This class provides a learned state estimator.
    This is trained with supervised learning, using only on-policy data
    collected in a rollout storage.
    predict() function provides state estimation for RL given the observation
    update() function optimizes for the nn params
    process_env_step() function stores values in a rollout storage
    """
    state_estimator: StateEstimatorNN

    def __init__(self,
                 state_estimator,    # nn module
                 learning_rate=1e-3,
                 num_mini_batches=1,
                 num_learning_epochs=1,
                 device='cpu',
                 **kwargs
                 ):

        # general parameters
        self.device = device
        self.learning_rate = learning_rate
        self.num_mini_batches = num_mini_batches
        self.num_learning_epochs = num_learning_epochs

        # SE storage
        self.transition = SERolloutStorage.Transition()
        self.storage = None

        # SE network and optimizer
        self.state_estimator = state_estimator
        self.state_estimator.to(self.device)
        self.optimizer = optim.Adam(self.state_estimator.parameters(),
                                    lr=learning_rate)
        self.SE_loss_fn = nn.MSELoss()

    def init_storage(self, num_envs, num_transitions_per_env,
                     obs_shape, se_shape):
        self.storage = SERolloutStorage(num_envs, num_transitions_per_env,
                                        obs_shape, se_shape,
                                        device=self.device)

    def predict(self, obs):
        return self.state_estimator.evaluate(obs)

    def process_env_step(self, obs, SE_targets):
        # Record the transition
        self.transition.SE_targets = SE_targets
        self.transition.observations = obs
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def update(self):
        """ Update the SE neural network weights via supervised learning """
        generator = self.storage.mini_batch_generator(self.num_mini_batches,
                                                      self.num_learning_epochs)
        mean_loss = 0
        for obs_batch, SE_target_batch in generator:

            SE_estimate_batch = self.state_estimator.evaluate(obs_batch)

            SE_loss = self.SE_loss_fn(SE_estimate_batch, SE_target_batch)
            self.optimizer.zero_grad()
            SE_loss.backward()
            self.optimizer.step()
            mean_loss += SE_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_loss /= num_updates
        self.storage.clear()

        return mean_loss

    def export(self, path):
        self.state_estimator.export(path)

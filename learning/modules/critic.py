import torch
import torch.nn as nn
from .utils import create_MLP
from .utils import RunningMeanStd


class Critic(nn.Module):
    def __init__(self,
                 num_obs,
                 hidden_dims,
                 activation="relu",
                 normalize_obs=True,
                 **kwargs):

        if kwargs:
            print("Critic.__init__ got unexpected arguments, "
                  "which will be ignored: "
                  + str([key for key in kwargs.keys()]))
        super().__init__()

        self.NN = create_MLP(num_obs, 1, hidden_dims, activation)

        self._normalize_obs = normalize_obs
        if self._normalize_obs:
            self.obs_rms = RunningMeanStd(num_obs)

    def evaluate(self, critic_observations):
        if self._normalize_obs:
            observations = self.norm_obs(critic_observations)
        return self.NN(observations).squeeze()

    def norm_obs(self, observation):
        with torch.no_grad():
            return self.obs_rms(observation) if self._normalize_obs else observation

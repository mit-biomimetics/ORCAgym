import torch
import torch.nn as nn
from torch.distributions import Normal
from .utils import RunningMeanStd
from .utils import create_MLP
from .utils import export_network


class Actor(nn.Module):
    def __init__(self,
                 num_obs,
                 num_actions,
                 hidden_dims,
                 activation="relu",
                 init_noise_std=1.0,
                 normalize_obs=True,
                 **kwargs):

        if kwargs:
            print("Actor.__init__ got unexpected arguments, "
                  "which will be ignored: "
                  + str([key for key in kwargs.keys()]))
        super().__init__()

        self._normalize_obs = normalize_obs
        if self._normalize_obs:
            self.obs_rms = RunningMeanStd(num_obs)

        self.num_obs = num_obs
        self.num_actions = num_actions
        self.NN = create_MLP(num_obs, num_actions, hidden_dims, activation)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @property
    def action_std(self):
        return self.std

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        if self._normalize_obs:
            observations = self.norm_obs(observations)
        mean = self.NN(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        if self._normalize_obs:
            observations = self.norm_obs(observations)
        actions_mean = self.NN(observations)
        return actions_mean

    def export(self, path):
        export_network(self.NN, "policy", path, self.num_obs)

    def norm_obs(self, observation):
        with torch.no_grad():
            return self.obs_rms(observation) if self._normalize_obs else observation

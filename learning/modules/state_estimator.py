import torch
import torch.nn as nn
from .utils import create_MLP
from .utils import export_network


class StateEstimatorNN(nn.Module):
    """ Set up a neural network for state-estimation

    Keyword arguments:
    num_inputs -- the number of observations used
    num_outputs -- the number of estimated states
    hidden_dims -- list of hidden layers and their sizes (default [256, 128])
    activation -- activation type (default 'elu')
    dropouts -- list of dropout rate for each layer (default None)
    """
    def __init__(self,  num_inputs, num_outputs, hidden_dims=[256, 128],
                 activation='elu', dropouts=None, **kwargs):
        if kwargs:
            print("StateEstimator.__init__ got unexpected arguments, "
                  "which will be ignored: "
                  + str([key for key in kwargs.keys()]))
        super().__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.NN = create_MLP(num_inputs, num_outputs, hidden_dims,
                             activation, dropouts)
        print(f"State Estimator MLP: {self.NN}")

    def evaluate(self, observations):
        return self.NN(observations)

    def export(self, path):
        export_network(self.NN, "state_estimator", path, self.num_inputs)

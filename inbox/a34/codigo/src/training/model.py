"""
DynamicMLP architecture definition.

Configurable Multi-Layer Perceptron that is dynamically built according
to the received hyperparameters (number of layers, neurons, activation).

Used across all training, optimization and evaluation notebooks.
"""

import torch.nn as nn


class DynamicMLP(nn.Module):
    """
    Multi-Layer Perceptron dynamically built from hyperparameters.

    Parameters
    ----------
    input_size : int
        Number of input features.
    output_size : int
        Number of output targets.
    num_layers : int
        Number of hidden layers.
    neurons : int
        Number of neurons per hidden layer.
    activation : str
        Activation function: 'ReLU' or 'Tanh'.

    Example
    -------
    >>> model = DynamicMLP(5, 2, num_layers=2, neurons=128, activation='ReLU')
    >>> # 5 -> 128 (ReLU) -> 128 (ReLU) -> 2
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_layers: int,
        neurons: int,
        activation: str = "ReLU",
    ):
        super(DynamicMLP, self).__init__()
        layers = []
        in_features = input_size
        act_fn = nn.ReLU() if activation == "ReLU" else nn.Tanh()

        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, neurons))
            layers.append(act_fn)
            in_features = neurons

        layers.append(nn.Linear(in_features, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

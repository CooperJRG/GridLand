import torch
import torch.nn as nn
import torch.nn.functional as f


class ValueNetwork(nn.Module):
    """
    Value network for reinforcement learning.

    Args:
        root_output (int): Size of the input layer.
        layers (list[int], optional): Sizes of hidden layers. Defaults to None.
    """

    def __init__(self, root_output, layers=None, activation=f.relu):
        super(ValueNetwork, self).__init__()

        self.fc_layers = []
        self.activation = activation

        if layers:
            self._validate_layers(layers)
            layers.insert(0, root_output)
            self.fc_layers = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
            self.value_head = nn.Linear(layers[-1], 1)
        else:
            self.value_head = nn.Linear(root_output, 1)

    @staticmethod
    def _validate_layers(layers):
        """
        Validate the sizes of hidden layers.

        Args:
            layers (list[int]): Sizes of hidden layers.

        Raises:
            ValueError: If any layer size is not a positive integer.
        """
        if not all(isinstance(i, int) and i > 0 for i in layers):
            raise ValueError("All layer sizes must be positive integers")

    def forward(self, x):
        """
        Forward pass of the value network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Value estimation.
        """
        for layer in self.fc_layers:
            x = self.activation(layer(x))

        value = self.value_head(x)

        return value

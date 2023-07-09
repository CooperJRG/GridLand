import torch
import torch.nn as nn
import torch.nn.functional as f


class PolicyNetwork(nn.Module):
    def __init__(self, root_output, layers=None, num_actions=5, activation=f.relu):
        """
    A PyTorch-based neural network model representing a policy for reinforcement learning.

    This network computes the policy (i.e., probability distribution over actions)
    given the processed state of an environment.

    Args:
        root_output (int): Number of nodes in the root layer.
        layers (list[int], optional): Sizes of hidden layers. Defaults to None.
            Each integer in this list represents the number of neurons in the corresponding hidden layer.
        num_actions (int, optional): Number of possible actions that the agent can take. Defaults to 5.
        activation (function, optional): Activation function to use between layers. Defaults to f.relu.
    """
        super(PolicyNetwork, self).__init__()

        self.fc_layers = []
        self.activation = activation
        if layers:
            self._validate_layers(layers)
            layers.insert(0, root_output)
            self.fc_layers = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
            self.policy_head = nn.Linear(layers[-1], num_actions)
        else:
            self.policy_head = nn.Linear(root_output, num_actions)

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
        Perform forward pass of the policy network, computing the policy for each action given the state `x`.

        Args:
            x (torch.Tensor): The tensor representing the state(s) of the environment.
                The tensor could represent a single state (1D tensor), or a batch of states (2D tensor).

        Returns:
            torch.Tensor: A tensor of the same length as `num_actions`, representing the probability
            distribution over actions. If `x` represents a batch of states, returns a 2D tensor where each row
            is the policy for a state.
        """

        for layer in self.fc_layers:
            x = self.activation(layer(x))

        x = self.policy_head(x)

        policy = f.softmax(x, dim=-1)

        return policy

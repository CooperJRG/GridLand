import torch
import torch.nn as nn
import torch.nn.functional as f

from PPO.policy_network import PolicyNetwork
from PPO.root_network import RootNetwork
from PPO.value_network import ValueNetwork


class DualNetwork(nn.Module):
    def __init__(self, device, input_dim, layers=None, use_cnn=False, kernel_sizes=None, strides=None, cnn_layers=None,
                 activation=f.relu, policy_layers=None, value_layers=None, num_actions=5):
        """
        Dual network model.

        Args:
            input_dim (int): Dimension of the input.
            layers (list[int], optional): Sizes of hidden layers. Defaults to None.
            use_cnn (bool, optional): Whether to use CNN layers. Defaults to False.
            kernel_sizes (list[int], optional): Sizes of kernel for CNN layers. Defaults to None.
            strides (list[int], optional): Sizes of stride for CNN layers. Defaults to None.
            cnn_layers (list[int], optional): Sizes of CNN layers. Defaults to None.
            activation (function, optional): Activation function. Defaults to f.relu.
            policy_layers (list[int], optional): Sizes of hidden layers in the policy network. Defaults to None.
            value_layers (list[int], optional): Sizes of hidden layers in the value network. Defaults to None.
        """
        super(DualNetwork, self).__init__()
        self.device = device
        self.root = RootNetwork(input_dim, layers, use_cnn, kernel_sizes, strides, cnn_layers, activation)
        self.policy = PolicyNetwork(self.root.output_size, policy_layers, num_actions, activation)
        self.value = ValueNetwork(self.root.output_size, value_layers, activation)
        self.root.to(self.device)
        self.policy.to(self.device)
        self.value.to(self.device)

    def forward(self, x):
        """
        Forward pass of the dual network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Value and policy tensors.
        """
        x = self.root(x)

        policy = self.policy(x)
        value = self.value(x)

        return policy, value

    def save_checkpoint(self, path):
        """
        Save the model parameters.

        Args:
            path (str): The path where to save the model parameters.
        """
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        """
        Load the model parameters.

        Args:
            path (str): The path from where to load the model parameters.
        """
        self.load_state_dict(torch.load(path))

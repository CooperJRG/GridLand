import functools
import operator

import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


class RootNetwork(nn.Module):
    """
    A neural network model for root classification.
    """

    def __init__(self, input_dim, layers=None, use_cnn=False, kernel_sizes=None, strides=None, cnn_layers=None,
                 activation=f.relu):
        """
        Initialize the RootNetwork model.

        Args:
            input_dim (Union[int, Tuple[int, int, int], List[Tuple[int, int, int]]]):
            Input dimension or a list of three tuples representing input dimensions.
            layers (list[int], optional): Sizes of hidden layers. Defaults to [64, 64].
            use_cnn (bool, optional): Whether to use convolutional layers. Defaults to False.
            kernel_sizes (list[tuple[int, int]], optional): Sizes of convolutional kernels.
            Defaults to [(3, 3), (3, 3), (3, 3)].
            strides (list[tuple[int, int]], optional): Sizes of convolutional strides.
            Defaults to [(3, 3), (2, 2), (1, 1)].
            cnn_layers (list[int], optional): Sizes of CNN layers. Defaults to [input_dim[0], 32, 64, 64].
            activation (function, optional): Activation function. Defaults to F.relu.
        """
        super(RootNetwork, self).__init__()

        self._validate_input_dim(input_dim, use_cnn)

        if layers is None:
            layers = [64, 64]
        if kernel_sizes is None:
            kernel_sizes = [(3, 3), (2, 2), (1, 1)]
        if strides is None:
            strides = [(3, 3), (2, 2), (1, 1)]
        if cnn_layers is None and use_cnn:
            cnn_layers = [input_dim[0], 32, 64, 64]
        elif use_cnn:
            cnn_layers.insert(0, input_dim[0])

        self._validate_layers(layers)
        self._validate_kernel_sizes(kernel_sizes)
        self._validate_strides(strides)
        if use_cnn:
            self._validate_cnn_layers(cnn_layers)
        self._validate_activation(activation)

        self.layers = layers
        self.use_cnn = use_cnn
        self.activation = activation
        self.output_size = layers[-1]

        # Check if CNN is required
        if self.use_cnn:
            self.cnn_layers = nn.ModuleList([
                nn.Conv2d(cnn_layers[i], cnn_layers[i + 1], kernel_size=kernel_sizes[i], stride=strides[i])
                for i in range(len(cnn_layers) - 1)
            ])
            x = torch.rand(1, *input_dim)
            for layer in self.cnn_layers:
                x = layer(x)
            num_features_before_fcnn = functools.reduce(operator.mul, list(x.shape))
            self.fc1 = nn.Linear(num_features_before_fcnn, layers[0])
        else:
            self.fc1 = nn.Linear(input_dim, layers[0])

        # Add any additional hidden layers
        self.fc_layers = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

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

    @staticmethod
    def _validate_kernel_sizes(kernel_sizes):
        """
        Validate the sizes of convolutional kernels.

        Args:
            kernel_sizes (list[tuple[int, int]]): Sizes of convolutional kernels.

        Raises:
            ValueError: If any kernel size is not a tuple of two positive integers.
        """
        if not all(isinstance(i, tuple) and len(i) == 2 and all(isinstance(j, int) and j > 0 for j in i) for i in
                   kernel_sizes):
            raise ValueError("All kernel sizes must be tuples of two positive integers")

    @staticmethod
    def _validate_strides(strides):
        """
        Validate the sizes of convolutional strides.

        Args:
            strides (list[tuple[int, int]]): Sizes of convolutional strides.

        Raises:
            ValueError: If any stride is not a tuple of two positive integers.
        """
        if not all(
                isinstance(i, tuple) and len(i) == 2 and all(isinstance(j, int) and j > 0 for j in i) for i in strides):
            raise ValueError("All strides must be tuples of two positive integers")

    @staticmethod
    def _validate_cnn_layers(cnn_layers):
        """
        Validate the sizes of CNN layers.

        Args:
            cnn_layers (list[int]): Sizes of CNN layers.

        Raises:
            ValueError: If any CNN layer size is not a positive integer.
        """
        if not all(isinstance(i, int) and i > 0 for i in cnn_layers):
            raise ValueError("All CNN layer sizes must be positive integers")

    @staticmethod
    def _validate_input_dim(input_dim, use_cnn):
        """
        Validate the input dimension.

        Args:
            input_dim (int): Input dimension.
            use_cnn (bool): Whether to use convolutional layers.

        Raises:
            ValueError: If the input dimension is not a positive integer or tuple of three ints.
        """
        if use_cnn:
            if not isinstance(input_dim, tuple) or not all(isinstance(i, int) and i > 0 for i in input_dim):
                raise ValueError("Input dimension must be a tuple of positive integers when using CNN")
        else:
            if not isinstance(input_dim, int) or input_dim <= 0:
                raise ValueError("Input dimension must be a positive integer when not using CNN")

    @staticmethod
    def _validate_activation(activation):
        """
        Validate the activation function.

        Args:
            activation (function): Activation function.

        Raises:
            ValueError: If the activation function is not callable.
        """
        if not callable(activation):
            raise ValueError("Activation must be a callable function")

    def _feature_size(self, input_dim):
        """
        Calculate the feature size for the fully connected layer.

        Args:
            input_dim (tuple[int, int, int]): Input dimension.

        Returns:
            int: Feature size.
        """
        return self._get_conv_output(input_dim)

    def _get_conv_output(self, shape):
        """
        Calculate the output shape of the convolutional layers.

        Args:
            shape (tuple[int, int, int]): Shape of the input.

        Returns:
            int: Output shape.
        """
        o = torch.zeros(1, *shape)
        for layer in self.cnn_layers:
            o = layer(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        Perform forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if self.use_cnn:
            for layer in self.cnn_layers:
                x = self.activation(layer(x))
            x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activation(x)

        # Apply activation function to each additional layer
        for layer in self.fc_layers:
            x = self.activation(layer(x))

        return x

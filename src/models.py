import torch
from torch import nn

class EnergyNet(nn.Module):
    """
    Defines the neural network architecture for the Energy-Based Model.
    This network takes an image as input and outputs a scalar energy value
    for each possible class.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list):
        """
        Initializes the EnergyNet.

        Args:
            input_dim (int): The flattened dimension of the input image (e.g., 28*28 for MNIST).
            output_dim (int): The number of output classes.
            hidden_dims (list): A list of integers specifying the sizes of hidden layers.
        """
        super(EnergyNet, self).__init__()

        layers = []
        in_features = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_features, h_dim))
            layers.append(nn.ELU()) # ELU activation function
            in_features = h_dim
        layers.append(nn.Linear(in_features, output_dim)) # Output layer
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor representing a batch of images.
                              Expected shape: (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with energy scores for each class.
                          Shape: (batch_size, num_classes).
        """
        # Flatten the input image from (batch_size, channels, height, width) to (batch_size, flattened_dim)
        x = x.view(x.size(0), -1)
        y_pred = self.net(x)
        return y_pred

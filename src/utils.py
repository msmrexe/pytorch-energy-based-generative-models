import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

def energy_gradient(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Calculates the gradient of the energy function with respect to the input x.

    Args:
        model (nn.Module): The EnergyNet model.
        x (torch.Tensor): Input tensor (e.g., a batch of images).

    Returns:
        torch.Tensor: Gradients of the energy with respect to x.
    """
    model.eval() # Set model to evaluation mode for gradient calculation
    x = x.clone().detach().requires_grad_(True) # Create a clone and ensure gradients are tracked

    # The energy is approximated by the LogSumExp of the model's output across classes.
    # Summing across the batch for a single scalar to backpropagate.
    energy = model(x).logsumexp(dim=1).sum()
    gradients = torch.autograd.grad(energy, x, retain_graph=False)[0]

    model.train() # Set model back to training mode
    return gradients

def langevin_dynamics_step(model: nn.Module, x: torch.Tensor, alpha: float, sigma: float) -> torch.Tensor:
    """
    Performs one step of Langevin dynamics.

    Args:
        model (nn.Module): The EnergyNet model.
        x (torch.Tensor): Current sample tensor.
        alpha (float): Step size (Langevin learning rate).
        sigma (float): Noise level.

    Returns:
        torch.Tensor: The new sample after one Langevin step.
    """
    grad = energy_gradient(model, x) # Calculate gradient
    epsilon = torch.randn_like(x) # Sample Gaussian noise

    # Langevin update rule
    x_new = x + (alpha * grad) + (sigma * epsilon)
    return x_new

def sample(model: nn.Module, eta: int, alpha: float, sigma: float,
           batch_size: int, image_size: int, device: str) -> torch.Tensor:
    """
    Generates new samples using Langevin dynamics.

    Args:
        model (nn.Module): The EnergyNet model.
        eta (int): Number of Langevin dynamics steps.
        alpha (float): Step size for Langevin dynamics.
        sigma (float): Noise level for Langevin dynamics.
        batch_size (int): Number of samples to generate.
        image_size (int): Size of the images (height and width).
        device (str): The device to generate samples on.

    Returns:
        torch.Tensor: Generated samples.
    """
    # Initialize samples from a uniform distribution [-1, 1]
    x = torch.empty((batch_size, 1, image_size, image_size), device=device).uniform_(-1, 1)

    # Detach x from its history to prevent backpropagating through Langevin steps
    x = x.detach()

    # Run Langevin dynamics for eta steps
    for _ in range(eta):
        x = langevin_dynamics_step(model, x, alpha, sigma)

    # Clamp the values to the valid range [-1, 1] for image display
    x = torch.clamp(x, -1, 1)
    return x

def visualize_real(loader: DataLoader, num_images: int = 16, save_path: str = None):
    """
    Visualizes a grid of real images from the dataset.

    Args:
        loader (DataLoader): DataLoader to fetch real images.
        num_images (int): Number of images to display in the grid.
        save_path (str, optional): Path to save the figure. If None, displays the plot.
    """
    data_iter = iter(loader)
    images, _ = next(data_iter)

    grid = torchvision.utils.make_grid(images[:num_images], nrow=int(np.sqrt(num_images)), normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Real Images")
    plt.axis("off")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close() # Close plot to prevent it from showing immediately
        logger.info(f"Real images visualized and saved to {save_path}")
    else:
        plt.show()
    logger.info("Real images displayed.")


def visualize_generated(model: nn.Module, eta: int, alpha: float, sigma: float,
                        batch_size: int, image_size: int, device: str,
                        num_images: int = 16, save_path: str = None):
    """
    Generates and visualizes a grid of images from the EBM.

    Args:
        model (nn.Module): The EnergyNet model.
        eta (int): Number of Langevin dynamics steps.
        alpha (float): Step size for Langevin dynamics.
        sigma (float): Noise level for Langevin dynamics.
        batch_size (int): Batch size to use for sampling.
        image_size (int): Size of the images (height and width).
        device (str): The device to generate samples on.
        num_images (int): Number of generated images to display in the grid.
        save_path (str, optional): Path to save the figure. If None, displays the plot.
    """
    model.eval() # Ensure model is in evaluation mode
    generated_images = sample(model, eta, alpha, sigma, batch_size, image_size, device).detach().cpu()

    grid = torchvision.utils.make_grid(generated_images[:num_images], nrow=int(np.sqrt(num_images)), normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.title("Generated Images")
    plt.axis("off")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close() # Close plot to prevent it from showing immediately
        logger.info(f"Generated images visualized and saved to {save_path}")
    else:
        plt.show()
    model.train() # Set model back to training mode
    logger.info("Generated images displayed.")

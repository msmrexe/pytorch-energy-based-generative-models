import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import os

from src.utils import sample, visualize_generated, visualize_real, energy_gradient, langevin_dynamics_step

logger = logging.getLogger(__name__)

def loss_function(model: nn.Module, eta: int, alpha: float, sigma: float,
                  x_real: torch.Tensor, y_pred_real: torch.Tensor, y_true: torch.Tensor,
                  image_size: int, device: str) -> torch.Tensor:
    """
    Calculates the combined discriminative and generative loss for the EBM.

    Args:
        model (nn.Module): The EnergyNet model.
        eta (int): Number of Langevin dynamics steps.
        alpha (float): Step size for Langevin dynamics.
        sigma (float): Noise level for Langevin dynamics.
        x_real (torch.Tensor): A batch of real images.
        y_pred_real (torch.Tensor): Model's predicted energy scores for real images.
        y_true (torch.Tensor): True labels for real images.
        image_size (int): Size of the images (height/width).
        device (str): The device to perform computations on.

    Returns:
        torch.Tensor: The total calculated loss.
    """
    # Discriminative loss: Cross-entropy for classification
    clf_loss = F.cross_entropy(y_pred_real, y_true)

    # Generative loss: E(x_real) - E(x_sample)
    # Generate negative samples using Langevin dynamics
    batch_size = x_real.size(0)
    x_sample = sample(model, eta, alpha, sigma, batch_size, image_size, device).detach()

    # Calculate energy for real and generated samples
    energy_real = model(x_real).logsumexp(dim=1) # LogSumExp over classes for each image
    energy_sample = model(x_sample).logsumexp(dim=1)

    # Maximize E(x_sample) - E(x_real) which means minimize -(E(x_sample) - E(x_real))
    # or equivalently minimize E(x_real) - E(x_sample)
    gen_loss = (energy_real - energy_sample).mean()

    total_loss = clf_loss + gen_loss
    return total_loss


def eval_ebm(model: nn.Module, loader: DataLoader, eta: int, alpha: float, sigma: float,
             image_size: int, device: str) -> tuple[float, float]:
    """
    Evaluates the EBM model on a given data loader.

    Args:
        model (nn.Module): The EnergyNet model.
        loader (DataLoader): DataLoader for evaluation data.
        eta (int): Number of Langevin dynamics steps.
        alpha (float): Step size for Langevin dynamics.
        sigma (float): Noise level for Langevin dynamics.
        image_size (int): Size of the images (height/width).
        device (str): The device to perform computations on.

    Returns:
        tuple[float, float]: A tuple containing average loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (x, y_true) in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
            x, y_true = x.to(device), y_true.to(device)

            y_pred = model(x)
            loss = loss_function(model, eta, alpha, sigma, x, y_pred, y_true, image_size, device)
            total_loss += loss.item()

            _, predicted_classes = torch.max(y_pred, 1)
            correct_predictions += (predicted_classes == y_true).sum().item()
            total_samples += y_true.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct_predictions / total_samples
    model.train() # Set model back to training mode
    return avg_loss, accuracy


def train_ebm(model: nn.Module, optimizer: optim.Optimizer, train_loader: DataLoader,
              test_loader: DataLoader, epochs: int, eta: int, alpha: float, sigma: float,
              device: str, visualization_freq: int, figure_save_path: str) -> tuple[list[float], list[float]]:
    """
    Trains the EBM model.

    Args:
        model (nn.Module): The EnergyNet model.
        optimizer (optim.Optimizer): Optimizer for model parameters.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for validation data.
        epochs (int): Number of training epochs.
        eta (int): Number of Langevin dynamics steps.
        alpha (float): Step size for Langevin dynamics.
        sigma (float): Noise level for Langevin dynamics.
        device (str): The device to perform computations on.
        visualization_freq (int): Frequency (in epochs) to visualize generated images.
        figure_save_path (str): Directory to save generated image figures.

    Returns:
        tuple[list[float], list[float]]: Lists of training and validation losses per epoch.
    """
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs} Training", position=0, leave=True) as pbar:
            for batch_idx, (x_real, y_true) in enumerate(train_loader):
                x_real, y_true = x_real.to(device), y_true.to(device)

                # Forward pass for real data
                y_pred_real = model(x_real)

                # Calculate total loss (discriminative + generative)
                loss = loss_function(model, eta, alpha, sigma, x_real, y_pred_real, y_true, x_real.shape[-1], device)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                pbar.set_postfix({"Train Loss": f"{epoch_train_loss / (batch_idx + 1):.4f}"})
                pbar.update(1)

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluate on validation set
        avg_val_loss, val_acc = eval_ebm(model, test_loader, eta, alpha, sigma, x_real.shape[-1], device)
        val_losses.append(avg_val_loss)

        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Visualize generated images periodically
        if (epoch + 1) % visualization_freq == 0:
            logger.info(f"Visualizing generated images after epoch {epoch+1}.")
            visualize_generated(
                model=model,
                eta=eta,
                alpha=alpha,
                sigma=sigma,
                batch_size=train_loader.batch_size, # Use loader's batch size for visualization
                image_size=x_real.shape[-1],
                device=device,
                save_path=os.path.join(figure_save_path, f"generated_epoch_{epoch+1}.png")
            )
    return train_losses, val_losses

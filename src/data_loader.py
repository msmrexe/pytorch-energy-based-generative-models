import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

def get_mnist_loaders(root: str = "./data", image_size: int = 28, batch_size: int = 64, device: str = "cpu") -> tuple[DataLoader, DataLoader]:
    """
    Loads the MNIST dataset and creates DataLoaders for training and testing.

    Args:
        root (str): Root directory for the dataset.
        image_size (int): Desired size of the images (height and width).
        batch_size (int): Number of samples per batch.
        device (str): The device to load data onto ('cpu', 'cuda', 'mps').

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing train_loader and test_loader.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize pixel values to range [-1, 1]
    ])

    try:
        trainset = MNIST(root=root, train=True, download=True, transform=transform)
        testset = MNIST(root=root, train=False, download=True, transform=transform)
    except Exception as e:
        logger.error(f"Error loading MNIST dataset: {e}")
        raise

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True if device == "cuda" else False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True if device == "cuda" else False)

    return trainloader, testloader

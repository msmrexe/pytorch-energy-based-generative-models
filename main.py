import torch
import torch.optim as optim
import logging
import argparse
import os

from config import Config
from src.data_loader import get_mnist_loaders
from src.models import EnergyNet
from src.training import train_ebm, eval_ebm, loss_function
from src.utils import visualize_real, visualize_generated, energy_gradient, langevin_dynamics_step, sample

# Setup logging
def setup_logging(log_file, log_level):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    # Suppress matplotlib font warnings
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description=Config.PROJECT_NAME)
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--langevin_steps', type=int, default=Config.LANGEVIN_STEPS, help='Number of Langevin dynamics steps (eta)')
    parser.add_argument('--langevin_alpha', type=float, default=Config.LANGEVIN_ALPHA, help='Step size for Langevin dynamics (alpha)')
    parser.add_argument('--langevin_sigma', type=float, default=Config.LANGEVIN_SIGMA, help='Noise level for Langevin dynamics (sigma)')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE, help='Learning rate for optimizer')
    parser.add_argument('--log_file', type=str, default=Config.LOG_FILE, help='Path to log file')
    parser.add_argument('--log_level', type=str, default=Config.LOG_LEVEL, help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    args = parser.parse_args()

    # Update config with parsed arguments
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LANGEVIN_STEPS = args.langevin_steps
    Config.LANGEVIN_ALPHA = args.langevin_alpha
    Config.LANGEVIN_SIGMA = args.langevin_sigma
    Config.LEARNING_RATE = args.lr
    Config.LOG_FILE = args.log_file
    Config.LOG_LEVEL = args.log_level

    # Initialize configuration and setup logging
    config_instance = Config()
    setup_logging(config_instance.LOG_FILE, config_instance.LOG_LEVEL)
    logger = logging.getLogger(__name__)

    logger.info(f"Configuration loaded: {vars(config_instance)}")
    logger.info(f"Current device: {config_instance.DEVICE}.")

    # 1. Data Loading
    logger.info("Downloading MNIST data...")
    trainloader, testloader = get_mnist_loaders(
        root=config_instance.DATA_ROOT,
        image_size=config_instance.IMAGE_SIZE,
        batch_size=config_instance.BATCH_SIZE,
        device=config_instance.DEVICE
    )
    logger.info(f"MNIST data loaded. Train: {len(trainloader.dataset)} samples, Test: {len(testloader.dataset)} samples.")

    # Visualize real images once
    visualize_real(trainloader, save_path=os.path.join(config_instance.FIGURE_SAVE_PATH, "real_images.png"))

    # 2. Model, Optimizer, and Loss
    model = EnergyNet(
        input_dim=config_instance.INPUT_DIM,
        output_dim=config_instance.NUM_CLASSES,
        hidden_dims=config_instance.HIDDEN_DIMS
    ).to(config_instance.DEVICE)
    logger.info(f"Model initialized: {model}")

    optimizer = optim.Adam(model.parameters(), lr=config_instance.LEARNING_RATE)

    # 3. Training
    logger.info("Starting EBM training...")
    train_losses, val_losses = train_ebm(
        model=model,
        optimizer=optimizer,
        train_loader=trainloader,
        test_loader=testloader,
        epochs=config_instance.EPOCHS,
        eta=config_instance.LANGEVIN_STEPS,
        alpha=config_instance.LANGEVIN_ALPHA,
        sigma=config_instance.LANGEVIN_SIGMA,
        device=config_instance.DEVICE,
        visualization_freq=config_instance.VISUALIZATION_FREQ,
        figure_save_path=config_instance.FIGURE_SAVE_PATH
    )
    logger.info("EBM training finished.")

    # 4. Final Visualization and Loss Plot
    visualize_generated(
        model=model,
        eta=config_instance.LANGEVIN_STEPS,
        alpha=config_instance.LANGEVIN_ALPHA,
        sigma=config_instance.LANGEVIN_SIGMA,
        batch_size=config_instance.BATCH_SIZE,
        image_size=config_instance.IMAGE_SIZE,
        device=config_instance.DEVICE,
        save_path=os.path.join(config_instance.FIGURE_SAVE_PATH, "final_generated_images.png")
    )

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, config_instance.EPOCHS + 1), train_losses, label="Training Loss")
    plt.plot(range(1, config_instance.EPOCHS + 1), val_losses, label="Validation Loss", linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(config_instance.FIGURE_SAVE_PATH, "loss_curves.png"))
    plt.show()
    logger.info("Loss curves plotted and saved.")

    # Optionally save the trained model
    torch.save(model.state_dict(), os.path.join(config_instance.MODEL_SAVE_PATH, "ebm_model.pth"))
    logger.info(f"Trained model saved to {os.path.join(config_instance.MODEL_SAVE_PATH, 'ebm_model.pth')}")

if __name__ == "__main__":
    main()

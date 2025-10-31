import os

class Config:
    # General settings
    PROJECT_NAME = "Energy-Based Generative Models"
    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    LOG_FILE = "ebm_training.log"
    LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # Dataset settings
    DATA_ROOT = "./data"
    IMAGE_SIZE = 28
    NUM_CLASSES = 10

    # Training settings
    EPOCHS = 15
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001

    # Model architecture settings
    INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE
    HIDDEN_DIMS = [512, 512, 512] # e.g., [512, 256, 128]

    # Langevin Dynamics parameters
    LANGEVIN_STEPS = 50  # eta
    LANGEVIN_ALPHA = 0.5 # alpha (increased for better convergence based on common EBM practices)
    LANGEVIN_SIGMA = 0.05 # sigma (increased for more effective exploration)

    # Visualization settings
    NUM_VISUALIZE_IMAGES = 16
    VISUALIZATION_FREQ = 5 # Visualize generated images every N epochs

    # Paths
    MODEL_SAVE_PATH = "./checkpoints/"
    FIGURE_SAVE_PATH = "./figures/"

    def __init__(self):
        os.makedirs(self.DATA_ROOT, exist_ok=True)
        os.makedirs(self.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(self.FIGURE_SAVE_PATH, exist_ok=True)

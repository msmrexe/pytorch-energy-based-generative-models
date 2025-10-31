# Energy-Based Generative Models

An implementation of Energy-Based Models (EBMs) leveraging Langevin dynamics for generative modeling on the MNIST dataset. This project explores learning a joint distribution of data and labels by minimizing an energy function, allowing for both discriminative classification and synthetic image generation. Developed as part of the Generative Models (M.S.) coursework.

## Features

 * **Modular Codebase:** Organized into logical modules for data handling, model definition, training, and utilities.
 * **Langevin Dynamics:** Implements the core mechanism for generating samples by navigating the energy landscape.
 * **Hybrid Loss Function:** Combines a discriminative cross-entropy loss with a generative term based on learned energy differences.
 * **MNIST Dataset Integration:** Configured to download, transform, and load the MNIST dataset for training and evaluation.
 * **Visualization Tools:** Includes functions to visualize real and generated images, aiding in model analysis.
 * **Configurable Hyperparameters:** All key training and model parameters are externalized for easy tuning.
 * **Comprehensive Logging:** Utilizes Python's `logging` module for detailed run-time information and error tracking, both to console and file.

## Core Concepts & Techniques

 * **Energy-Based Models (EBMs):** A class of generative models that define a scalar energy function, where lower energy corresponds to higher probability data points.
 * **Langevin Dynamics:** A Markov Chain Monte Carlo (MCMC) method used for sampling from probability distributions defined by an energy function. It involves iterative steps with gradient ascent on the energy function and added Gaussian noise.
 * **Log-Sum-Exp Trick:** Employed in the energy calculation to improve numerical stability when dealing with probabilities, especially in the context of approximating the partition function.
 * **Generative and Discriminative Learning:** The model is trained with a hybrid objective that allows it to simultaneously perform classification and generate new data samples.
 * **PyTorch Framework:** Leveraging PyTorch for efficient tensor operations, automatic differentiation, and neural network construction.

---

## How It Works

This project trains a neural network to model the data distribution of MNIST by learning an **energy function**. This function assigns a low scalar "energy" value to realistic (image, label) pairs and a high energy to unrealistic pairs.

### 1. Theoretical Foundation: Energy-Based Models (EBMs)

An EBM defines a probability distribution $p_{\theta}(x)$ through an energy function $E_{\theta}(x)$, where $\theta$ represents the parameters of our neural network. The relationship is defined by the Gibbs distribution:

$$p_{\theta}(x) = \frac{e^{-E_{\theta}(x)}}{Z(\theta)}$$

* $E_{\theta}(x)$: The energy function. If $x$ is a realistic image, $E_{\theta}(x)$ is low. If $x$ is noise, $E_{\theta}(x)$ is high.
* $Z(\theta) = \int_{x'} e^{-E_{\theta}(x')} dx'$: The partition function. This is a normalization constant that ensures $p_{\theta}(x)$ sums to 1.

**The Training Problem:** The partition function $Z(\theta)$ is intractable to compute for high-dimensional data like images, as it requires integrating over all possible images.

**The Training Solution (Contrastive Divergence):** We don't need to compute $Z(\theta)$ to train the model. We can train by maximizing the log-likelihood of our real data $x$. The gradient of the log-likelihood is:

$$\nabla_{\theta} \log p_{\theta}(x) = \mathbb{E}\_{x' \sim p_{\theta}}[\nabla_{\theta} E_{\theta}(x')] - \nabla_{\theta} E_{\theta}(x)$$

This gradient gives us a simple, two-part training rule:
1.  **"Positive" Phase ( $-\nabla_{\theta} E_{\theta}(x)$ ):** For a **real image** $x$ from our dataset, we "push down" on its energy. We adjust $\theta$ to make $E_{\theta}(x)$ smaller.
2.  **"Negative" Phase ( $\mathbb{E}\_{x' \sim p_{\theta}}[\nabla_{\theta} E_{\theta}(x')]$ ):** For a **"negative" sample** $x'$ drawn from our model's own distribution $p_{\theta}$, we "push up" on its energy. We adjust $\theta$ to make $E_{\theta}(x')$ larger.

This process is "contrastive" because it trains the model to distinguish between real data (which it makes "cheap" energetically) and fake data (which it makes "expensive"). This sculpts the energy landscape, creating low-energy "valleys" around the real data points.

### 2. Implementation: Joint Energy Model and Langevin Dynamics

This project implements a **joint energy model** $E_{\theta}(x, y)$, which learns the energy of an image $x$ *and* its label $y$.

#### **Model Input/Output**
* **Model:** `EnergyNet`, a simple Multi-Layer Perceptron (MLP).
* **Input ($x$):** A flattened $28 \times 28$ MNIST image tensor.
* **Output ($s$):** A vector of 10 scores, $s = [s_0, s_1, ..., s_9]$. Each score $s_i$ represents the model's energy prediction for the input image $x$ paired with class $i$.

#### **Training Algorithm (The Full Loop)**
The model is trained with a hybrid objective, combining a classification loss and a generative loss.

1.  **Fetch Real Data:** Get a batch of $(x_{\text{real}}, y_{\text{true}})$ from the MNIST training loader.
2.  **Forward Pass (Real Data):** Calculate the output scores for the real images: $s_{\text{real}} = \text{Model}(x_{\text{real}})$.
3.  **Calculate Discriminative Loss ($\mathcal{L}_{\text{clf}}$):**
    * We treat the 10 output scores as standard classification logits.
    * $\mathcal{L}\_{\text{clf}} = \text{CrossEntropyLoss}(s_{\text{real}}, y_{\text{true}})$.
    * This loss "pushes down" the energy $s_i$ for the *correct* class $i$ relative to all other classes, teaching the model to be a good classifier.

4.  **Generate Negative Samples ($x_{\text{sample}}$) via Langevin Dynamics:**
    * To get the $x'$ samples for the "negative" phase, we use **Langevin Dynamics**. This is an MCMC algorithm that generates samples by following the energy landscape *downhill* (with some noise for exploration).
    * **a. Start:** Initialize a batch of random images $x_0 \sim \text{Uniform}(-1, 1)$.
    * **b. Iterate:** For $t=0$ to $\eta-1$:
        * i. Calculate the energy of the current image, $E(x_t) = \text{LogSumExp}(s_0, ..., s_9)$ where $s = \text{Model}(x_t)$. The LogSumExp marginalizes out the 10 class energies to get a single energy for the image $x_t$.
        * ii. Get the gradient of the energy w.r.t. the image: $g_t = \nabla_{x_t} E(x_t)$.
        * iii. Sample Gaussian noise: $\epsilon_t \sim \mathcal{N}(0, I)$.
        * iv. Update the image by taking a step *down* the energy gradient (plus noise):
            $$x_{t+1} = x_t - \alpha \cdot g_t + \sigma \cdot \epsilon_t$$
    * **c. Finish:** The resulting image $x_{\eta}$ is our "negative" sample $x_{\text{sample}}$.

5.  **Calculate Generative Loss ($\mathcal{L}_{\text{gen}}$):**
    * Forward pass the negative samples: $s_{\text{sample}} = \text{Model}(x_{\text{sample}})$.
    * Get the energy for real and negative samples:
        * $E_{\text{real}} = \text{LogSumExp}(s_{\text{real}})$
        * $E_{\text{sample}} = \text{LogSumExp}(s_{\text{sample}})$
    * $\mathcal{L}\_{\text{gen}} = E_{\text{real}}.mean() - E_{\text{sample}}.mean()$
    * Minimizing this loss *pushes down* on $E_{\text{real}}$ (the positive phase) and *pushes up* on $E_{\text{sample}}$ (the negative phase).

6.  **Optimization:**
    * Calculate the total loss: $\mathcal{L} = \mathcal{L}\_{\text{clf}} + \mathcal{L}_{\text{gen}}$.
    * Perform backpropagation and update the model parameters $\theta$ using $\nabla_{\theta} \mathcal{L}$.

7.  **Repeat:** This entire process is repeated for each batch and epoch.

---

## Project Structure

```
pytorch-energy-based-generative-models/
├── .gitignore           # Specifies intentionally untracked files to ignore
├── LICENSE              # MIT License for the project
├── README.md            # Project overview, features, how it works, and usage instructions
├── requirements.txt     # Project dependencies
├── main.py              # Main script to run the EBM training and evaluation. Orchestrates all components
├── config.py            # Configuration file for hyperparameters and settings
└── src/                 # Source code directory
│   ├── models.py        # Defines the EnergyNet neural network architecture
│   ├── data_loader.py   # Handles dataset loading, transformations, and DataLoader setup
│   ├── training.py      # Contains functions for the EBM's loss calculation, training loop, and evaluation
│   └── utils.py         # Implements Langevin dynamics, sampling procedures, and visualization tools
└── notebooks/
    └── run_ebm.ipynb    # A Jupyter notebook to demonstrate how to run the EBM training and visualize results
```

## How to Use

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/msmrexe/pytorch-energy-based-generative-models.git
    cd pytorch-energy-based-generative-models
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    pip install torch torchvision numpy matplotlib tqdm
    ```

3.  **Run the EBM Training Script:**
    You can run the main training script directly from the command line. You can override default parameters defined in `config.py` using command-line arguments.

    ```bash
    python main.py --epochs 20 --batch_size 128 --langevin_steps 60 --langevin_alpha 0.5 --langevin_sigma 0.05
    ```

      * `--epochs`: Number of training epochs. Default: `15`.
      * `--batch_size`: Batch size for data loaders. Default: `64`.
      * `--langevin_steps`: Number of Langevin dynamics steps (`eta`). Default: `50`.
      * `--langevin_alpha`: Step size for Langevin dynamics (`alpha`). Default: `0.1`.
      * `--langevin_sigma`: Noise level for Langevin dynamics (`sigma`). Default: `0.01`.
      * `--lr`: Learning rate for the optimizer. Default: `0.001`.
      * `--log_file`: Path to the log file. Default: `ebm_training.log`.

4.  **Explore with Jupyter Notebook:**
    For an interactive experience and step-by-step execution, you can use the provided Jupyter notebook.

    ```bash
    jupyter notebook notebooks/run_ebm.ipynb
    ```

    Follow the instructions within the notebook to load data, train the model, and visualize results.

### Example Usage / Test the System

When running `main.py`, you will see real-time training progress in your console, including epoch number, training loss, validation loss, and validation accuracy. Every 5 epochs, the script will also display a grid of generated images.

<!---
Example console output during training:

```
INFO:root:Configuration loaded: {'epochs': 15, 'batch_size': 64, 'image_size': 28, 'num_classes': 10, 'langevin_steps': 50, 'langevin_alpha': 0.5, 'langevin_sigma': 0.05, 'learning_rate': 0.001, 'log_file': 'ebm_training.log'}
INFO:root:Current device: cuda.
Downloading MNIST data...
INFO:root:MNIST data loaded. Train: 60000 samples, Test: 10000 samples.
INFO:root:Model initialized: EnergyNet(
  (net): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ELU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ELU()
    (4): Linear(in_features=512, out_features=512, bias=True)
    (5): ELU()
    (6): Linear(in_features=512, out_features=10, bias=True)
  )
)
Epoch 1/15: 100%|██████████| 938/938 [01:29<00:00, 10.51it/s, Train Loss=1.1440]
INFO:root:Epoch 1/15 | Train Loss: 1.1440 | Val Loss: -2.6497 | Val Acc: 0.8137
Epoch 2/15: 100%|██████████| 938/938 [01:25<00:00, 10.91it/s, Train Loss=0.5118]
INFO:root:Epoch 2/15 | Train Loss: 0.5118 | Val Loss: -1.7308 | Val Acc: 0.9194
...
INFO:root:Epoch 5/15 | Train Loss: 0.4829 | Val Loss: 7.1384 | Val Acc: 0.9478
INFO:root:Visualizing generated images after epoch 5.
```

Upon reaching epoch 5 (and subsequent multiples of 5), a matplotlib window will pop up showing the generated images:

At the end of training, a plot of training and validation loss over epochs will be displayed:

--->

-----

## Author

Feel free to connect or reach out if you have any questions!

  * **Maryam Rezaee**
  * **GitHub:** [@msmrexe](https://github.com/msmrexe)
  * **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)

-----

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.

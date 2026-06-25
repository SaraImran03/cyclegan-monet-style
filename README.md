# Claude Monet Style Transfer using CycleGAN

This repository implements a **Cycle-Consistent Generative Adversarial Network (CycleGAN)** for unpaired image-to-image translation. The model is trained to transform real-world photographs into impressionist-style paintings matching the unique artistic signature of **Claude Monet**.

Unpaired style transfer is a challenging computer vision task because matching pairs of photos and paintings do not exist. CycleGAN overcomes this limitation by training two generator-discriminator pairs simultaneously under a cycle-consistency constraint.

---

## 🎨 Model Architecture

The model architecture consists of two Generators and two Discriminators working in parallel to translate images between Domain $X$ (real photos) and Domain $Y$ (Monet paintings).

```
          Generator G (X -> Y)
  [Photo X] ───────────────────► [Generated Monet Y']
      │                                     │
      │                                     │ Generator F (Y' -> X'')
      │                                     ▼
      └────────────────────────────── [Reconstructed X'']
            L1 Cycle Consistency Loss
```

### 1. Generators ($G: X \to Y$ and $F: Y \to X$)
* **Architecture**: A Convolutional Neural Network (CNN) employing downsampling (encoder), bottleneck layers (residual layers), and upsampling (decoder/transposed convolutions) to preserve spatial features while translating styles.
* **Layers**:
  * **Downsampling**: 2D Convolutional layers (`Conv2d`) with stride for feature extraction and dimension reduction.
  * **Upsampling**: Transposed Convolutional layers (`ConvTranspose2d`) for reconstructing spatial resolution.
* **Activation Functions**:
  * `LeakyReLU` activations are used in intermediate layers for gradient flow.
  * A `Tanh` activation function is used in the final output layer to scale output pixel values to $[-1, 1]$.

### 2. Discriminators ($D_X$ and $D_Y$)
* **Architecture**: Critic networks based on PatchGAN that classify overlapping image patches as real or fake. This encourages the generator to focus on local high-frequency details (such as brushstroke textures).
* **Layers**: Standard 2D Convolutional layers reducing the spatial dimension to a patch grid.
* **Activation Functions**: `LeakyReLU` activations throughout the hidden layers.
* **Output**: A single probability score grid (values close to 1 represent real images, values close to 0 represent generated/fake images).

---

## ⚖️ Loss Formulation

To ensure stable training and structural preservation, the networks are trained with a composite objective function:

### 1. Least Squares Adversarial Loss ($\mathcal{L}_{GAN}$)
Instead of the standard log-likelihood loss, a Mean Squared Error (MSE) loss is used for the adversarial objective to stabilize training:

$$\mathcal{L}_{GAN}(G, D_Y, X, Y) = \mathbb{E}_{y \sim p_{\text{data}}(y)}[(D_Y(y) - 1)^2] + \mathbb{E}_{x \sim p_{\text{data}}(x)}[D_Y(G(x))^2]$$

$$\mathcal{L}_{GAN}(F, D_X, Y, X) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[(D_X(x) - 1)^2] + \mathbb{E}_{y \sim p_{\text{data}}(y)}[D_X(F(y))^2]$$

### 2. Cycle Consistency Loss ($\mathcal{L}_{\text{cyc}}$)
To prevent mode collapse and force the model to preserve structural context (so a house remains a house, just painted in Monet's style), L1 cycle-consistency loss is applied:

$$\mathcal{L}_{\text{cyc}}(G, F) = \mathbb{E}_{x \sim p_{\text{data}}(x)}\left[\|F(G(x)) - x\|_1\right] + \mathbb{E}_{y \sim p_{\text{data}}(y)}\left[\|G(F(y)) - y\|_1\right]$$

### 3. Total Objective Function
$$\mathcal{L}(G, F, D_X, D_Y) = \mathcal{L}_{GAN}(G, D_Y, X, Y) + \mathcal{L}_{GAN}(F, D_X, Y, X) + \lambda \mathcal{L}_{\text{cyc}}(G, F)$$

Where the cycle consistency weight is set to $\lambda = 10$.

---

## 📊 Dataset & Setup

The model was trained on the Kaggle **GANs Getting Started** dataset:
* **Monet Paintings (Domain Y)**: 300 images
* **Real Photographs (Domain X)**: 2,000 images
* **Test Dataset**: 400 photos for inference

---

## 🔬 Experimental Results

Two training configurations were compared to study the impact of hyperparameter choices on style-transfer fidelity and texture details.

| Hyperparameter | Experimental Setup 01 (Baseline) | Experimental Setup 02 (Optimized) |
| :--- | :--- | :--- |
| **Epochs** | 10 epochs | **50 epochs** |
| **Batch Size** | 6 | **1** |
| **Learning Rate** | $2 \times 10^{-4}$ | $2 \times 10^{-4}$ |
| **FID Score** | **335.95** | **181.71** (Lower is better) |

### 📈 Comparison & Observations

* **Setup 01**: Due to a larger batch size (6) and limited epochs (10), the model fails to capture detailed brushstrokes. The resulting images are highly blurred and exhibit flat color grading similar to a sepia or vignette photo filter.
* **Setup 02**: Increasing training length to 50 epochs and reducing the batch size to 1 yields significant improvements. The model successfully captures Monet's signature impressionist painting style, including distinct brushstroke textures, color blending, and paint layering. The FID score improved dramatically by **45.9%** (down to **181.71**).

---

## 🚀 Getting Started

### 📋 Prerequisites
Install the required PyTorch stack and plotting libraries:
```bash
pip install torch torchvision numpy matplotlib pillow
```

### 🧠 Inference Usage
You can run inference using the trained generator weights `best_monet_generator.pth` to style-transfer your own images:

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# 1. Define Model and Load Weights
# (Ensure your Generator class matching CycleGAN architecture is imported/defined)
generator = Generator()
generator.load_state_dict(torch.load('models/experiment_02/best_monet_generator.pth', map_location='cpu'))
generator.eval()

# 2. Preprocess Image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

img = Image.open('path_to_your_photo.jpg').convert('RGB')
input_tensor = transform(img).unsqueeze(0)

# 3. Perform Style Transfer
with torch.no_grad():
    output_tensor = generator(input_tensor)

# 4. Denormalize and Save
output_image = (output_tensor.squeeze(0) * 0.5) + 0.5
output_image = transforms.ToPILImage()(output_image.clamp(0, 1))
output_image.save('monet_style_output.jpg')
```

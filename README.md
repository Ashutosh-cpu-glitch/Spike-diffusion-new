# SpikeDiffusion  
### Diffusion-Inspired Gaussian Image Denoising using Spiking Neural Networks

SpikeDiffusion is an experimental deep learning project that explores whether Spiking Neural Networks (SNNs) can learn diffusion-style Gaussian image denoising on the MNIST dataset.

The project combines a lightweight residual spiking architecture with a simplified diffusion process to study how spike-based temporal computation behaves in image restoration tasks.

Rather than focusing on large-scale image generation, this work investigates the feasibility of using spike-driven neural dynamics for noise prediction and image reconstruction.

---

# Project Motivation

Diffusion models have recently shown strong performance in image restoration and generative modeling by learning to reverse progressive noise corruption. However, most modern diffusion systems rely on computationally intensive deep neural networks and transformer-based architectures.

In contrast, Spiking Neural Networks process information through discrete spike activity and temporal membrane dynamics, making them biologically inspired and potentially more computationally efficient in neuromorphic settings.

This project explores the intersection of these two areas by asking a simple research-oriented question:

> Can a lightweight residual spiking neural network learn diffusion-style Gaussian denoising behavior?

---

# Objectives

The main objectives of this work are:

- Implement Gaussian forward diffusion corruption
- Design a residual spiking convolutional architecture
- Train the model to predict injected Gaussian noise
- Reconstruct cleaner images from noisy inputs
- Evaluate denoising performance quantitatively and visually
- Explore diffusion learning behavior in spike-based neural systems

---

# Methodology

## Forward Diffusion

The forward diffusion process progressively corrupts an image by adding Gaussian noise over multiple timesteps.

At timestep `t`:

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon
$$

where:

- $x_0$ is the original image  
- $\epsilon$ is Gaussian noise  
- $\bar{\alpha}_t$ controls accumulated noise intensity  

As diffusion timesteps increase, image quality gradually degrades.

---

## Spiking Residual Architecture

The denoising model is built using:

- Convolutional feature extraction
- Residual learning
- Leaky Integrate-and-Fire (LIF) neurons
- Group normalization
- Temporal spike simulation
- Sinusoidal timestep embeddings

The architecture predicts the Gaussian noise added during forward diffusion.

### Key Design Choices

- Residual convolution blocks for stable optimization
- Spike-based temporal dynamics using `snntorch`
- Lightweight channel configuration for efficient experimentation
- Time-conditioned processing using sinusoidal embeddings

---

## Training Strategy

During training:

1. Random diffusion timesteps are sampled
2. Gaussian noise is injected into MNIST images
3. The noisy image is passed through the spiking network
4. The model predicts the injected noise
5. Mean Squared Error (MSE) loss is minimized

### Optimization Setup

- Optimizer: AdamW
- Scheduler: CosineAnnealingLR
- Loss Function: MSE Loss

---

# Dataset

The project uses the MNIST handwritten digit dataset.

- Training Samples: 60,000
- Test Samples: 10,000
- Image Size: 28 × 28 grayscale

MNIST was chosen to keep the study computationally manageable while focusing on the behavior of the spiking diffusion framework itself.

---

# Technologies Used

```python
Python
PyTorch
snntorch
Torchvision
Matplotlib
tqdm
```

---

# Experimental Results

After training for 10 epochs, the model demonstrated measurable denoising capability on unseen MNIST test samples.

### Evaluation Results

```text
Average Noisy MSE    : 0.066020
Average Denoised MSE : 0.004450
Improvement          : 93.26%
```

The notebook also includes:

- Forward diffusion visualization
- Denoising reconstruction examples
- Training loss analysis
- Qualitative comparison between original, noisy, and reconstructed images

---

# Key Observations

- The residual spiking network successfully learned Gaussian noise prediction
- Denoised outputs were significantly closer to original images than noisy inputs
- Residual convolutional processing helped preserve spatial structure
- Temporal spike dynamics participated effectively in denoising behavior

---

# Limitations

This work is intentionally exploratory and has several limitations:

- Focused on denoising rather than full image generation
- Uses a lightweight architecture
- Reverse diffusion sampling is simplified
- Computational cost increases due to temporal spike simulation
- Experiments are limited to MNIST-scale data

---

# Future Directions

Possible extensions of this work include:

- Multi-step reverse diffusion sampling
- Spiking U-Net architectures
- Hybrid ANN-SNN diffusion systems
- Attention-based spike diffusion models
- Larger and more complex datasets
- Neuromorphic hardware deployment

---

# Research Perspective

This project was developed as an exploratory study at the intersection of:

- Spiking Neural Networks
- Diffusion-based learning
- Temporal neural computation
- Image restoration systems

The goal was not to reproduce state-of-the-art diffusion performance, but to investigate whether spike-based neural processing can meaningfully participate in diffusion-inspired denoising tasks.

---

# Repository Structure

```text
SpikeDiffusion/
│
├── Spike_Diffusion_Denoising.ipynb
├── README.md
├── LICENSE
├── requirements.txt
└── results/
    ├── forward_diffusion.png
    ├── training_loss.png
    └──denoising_result.png
```

---

# Conclusion

This study demonstrates that a lightweight residual spiking neural network can learn diffusion-style Gaussian image denoising behavior on MNIST images.

The results suggest that spike-based temporal computation can be integrated with diffusion-inspired learning objectives for image restoration tasks, while also opening possibilities for future exploration in neuromorphic diffusion systems.

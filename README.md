# SpikingDiffusion-Neuromorphic-Modeling

This project explores a diffusion-inspired generative modeling prototype using Spiking Neural Networks (SNNs) for image and audio data. The goal is to investigate whether spike-based neural systems can learn to represent, reconstruct, and iteratively refine signals under stochastic corruption.

---

## Project Overview

This repository implements a prototype diffusion-style framework using SNNs, along with supporting experiments in image reconstruction and spike-based signal processing.

The project is divided into three main components:

* Spiking Autoencoder (MNIST)
* Spiking Diffusion Denoising Model (MNIST)
* Spiking Diffusion for Audio (Exploratory Prototype)

The emphasis is on:

* Understanding spike-based representations
* Exploring denoising as a generative principle
* Evaluating the feasibility of neuromorphic generative modeling

---

## Repository Structure

```
SpikingDiffusion/
│
├── core/
│   ├── __init__.py
│   ├── models.py
│   ├── diffusion.py
│   ├── utils.py
│
├── 1_spiking_autoencoder_mnist.ipynb
├── 2_spiking_diffusion_denoising_mnist.ipynb
├── 3_spiking_diffusion_audio.ipynb
│
└── README.md
└── requirement.txt
└── LICENSE
```

---

## Methods and Concepts

### 🔹 Spiking Neural Networks (SNNs)

The models are implemented using Leaky Integrate-and-Fire (LIF) neurons via snnTorch, enabling temporal spike-based computation.

### 🔹 Rate Coding

Static inputs (images and signals) are converted into spike trains using rate coding, where intensity is represented by firing frequency.

### 🔹 Diffusion-style Modeling

A simplified diffusion formulation is used:

x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon

* Forward process: progressively corrupt data with noise
* Reverse process: train a model to approximate a reverse mapping from corrupted inputs

### 🔹 Poisson Spike Noise

Noise is modeled using a Poisson-like process, inspired by stochastic spike generation.

---

## Notebooks

### 1️⃣ Spiking Autoencoder (MNIST)

* Encodes images into spike trains using rate coding
* Reconstructs images from spike activity
* Demonstrates that SNNs can learn to reconstruct input data from spike-based representations

**Key idea:**
Spike rates encode information, and temporal averaging recovers the signal.

---

### 2️⃣ Spiking Diffusion Denoising (MNIST)

* Implements a diffusion-style corruption process
* Trains a spiking neural network to predict injected noise
* Demonstrates partial denoising behavior

**Key Result:**
Quantitative evaluation shows:
MSE(Noisy vs Original) > MSE(Denoised vs Original)

This indicates that the model learns to partially reverse corruption under a diffusion-style objective.

**Note:**
Full generative sampling from pure noise is computationally expensive, especially with SNNs. This experiment focuses on validating the denoising objective.

---

### 3️⃣ Spiking Diffusion for Audio (Exploratory Prototype)

* Encodes a synthetic waveform into spike trains using rate coding
* Trains a spiking neural network to map noisy inputs to structured signal outputs
* Applies an iterative update procedure to generate a waveform from random initialization

This experiment explores whether spike-based models can learn signal refinement behavior under a diffusion-inspired training setup.

---

## Evaluation

Evaluation is task-dependent:

* **Image-based experiments:**
  Mean Squared Error (MSE) is used to compare reconstructed or denoised outputs with original inputs

* **Audio experiments:**
  Signal-level statistics are computed, including power, variance, spectral energy, and zero-crossing rate

These metrics provide insight into reconstruction accuracy and signal structure.

---

## Limitations

* Diffusion-style models are computationally expensive
* Training SNNs with surrogate gradients is non-trivial
* The current implementation is a prototype, not a fully converged generative system
* Generated outputs, particularly in audio, remain limited in complexity

---

## Key Takeaways

* SNNs can encode and reconstruct signals using spike-based representations
* Diffusion-style denoising can be implemented with spiking neurons
* Quantitative results indicate consistent reduction in reconstruction error after denoising
* This project provides a foundation for exploring neuromorphic generative modeling

---
---

## Requirements

```bash
pip install -r requirements.txt
```
## Acknowledgement
This project is an exploratory research prototype aimed at understanding the intersection of:

-Neuromorphic computing

-Generative modeling

-Spike-based information processing

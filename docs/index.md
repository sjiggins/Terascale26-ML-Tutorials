# Generative Modeling Tutorials

Welcome to the comprehensive tutorial series on **Generative Modeling**! This repository contains hands-on, educational tutorials designed to help you understand and implement modern generative models from scratch.

## Overview

This tutorial series covers the fundamentals of generative modeling, from classical diffusion models to modern flow matching techniques. Each tutorial includes:

- Complete working code implementations
- Step-by-step Jupyter notebooks
- Detailed mathematical explanations
- Visualizations and animations
- Exercises and experiments

## Tutorials

### Tutorial 1: Denoising Diffusion Probabilistic Models (DDPM)

**Status:** Available

Learn the foundations of diffusion models by implementing DDPM from scratch.

**Topics Covered:**
- Forward diffusion process (adding noise)
- Reverse diffusion process (denoising)
- Training a score-based model
- Sampling with various schedulers

[Start Tutorial 1](tutorials/tutorial-1.md){ .md-button .md-button--primary }

---

### Tutorial 2: Flow Matching

**Status:** Available

Explore flow matching as an alternative to stochastic diffusion models.

**Topics Covered:**
- Probability paths (linear vs variance preserving)
- Velocity field learning
- ODE-based sampling
- Comparison with DDPM

[Start Tutorial 2](tutorials/tutorial-2.md){ .md-button .md-button--primary }

---

### Tutorial 3: Advanced Topics

**Status:** Coming Soon

Dive deeper into advanced generative modeling techniques.

**Planned Topics:**
- Conditional generation
- Classifier-free guidance
- Latent diffusion models

[Coming Soon](tutorials/tutorial-3.md){ .md-button }

---

### Tutorial 4: Score-Based Models

**Status:** Coming Soon

Explore score-based generative modeling and SDEs.

**Planned Topics:**
- Score matching
- Stochastic differential equations
- Advanced sampling techniques

[Coming Soon](tutorials/tutorial-4.md){ .md-button }

---

### Tutorial 5: Applications

**Status:** Coming Soon

Apply generative models to real-world problems.

**Planned Topics:**
- Image generation
- Audio synthesis
- Molecular design

[Coming Soon](tutorials/tutorial-5.md){ .md-button }

---

## Quick Start

!!! tip "New to these tutorials?"
    1. Start with [Installation](installation.md) to set up your environment
    2. Follow the [Getting Started](getting-started.md) guide to clone and configure
    3. Begin with [Tutorial 1](tutorials/tutorial-1.md) for foundations

## Learning Path

```mermaid
graph LR
    A[Setup Environment] --> B[Tutorial 1: DDPM]
    B --> C[Tutorial 2: Flow Matching]
    C --> D[Tutorial 3: Advanced Topics]
    D --> E[Tutorial 4: Score-Based]
    D --> F[Tutorial 5: Applications]
```

## Prerequisites

- **Python 3.9+**: Modern Python installation
- **PyTorch 2.0+**: Deep learning framework
- **Basic ML Knowledge**: Understanding of neural networks and gradient descent
- **Linear Algebra**: Matrix operations, eigenvalues
- **Calculus**: Derivatives, gradients, probability

## Features

- **Hands-On Learning**: Complete working implementations you can run and modify
- **Interactive Notebooks**: Jupyter notebooks with step-by-step explanations
- **Visualizations**: Plots and animations to build intuition
- **Mathematical Rigor**: Clear mathematical explanations with LaTeX
- **Modular Code**: Clean, well-documented code you can reuse
- **Exercises**: Practice problems to reinforce learning

## System Requirements

=== "Minimum"
    - **CPU**: 2+ cores
    - **RAM**: 4 GB
    - **Storage**: 2 GB free space
    - **OS**: Linux, macOS, or Windows 10+

=== "Recommended"
    - **CPU**: 4+ cores or GPU
    - **RAM**: 8 GB+
    - **Storage**: 5 GB free space
    - **OS**: Linux or macOS
    - **GPU**: NVIDIA GPU with CUDA support (optional but faster)

## Support

If you encounter issues:

1. Check the [FAQ](faq.md)
2. Review [Troubleshooting](troubleshooting.md)
3. Open an issue on [GitHub](https://github.com/yourusername/generative-tutorials/issues)

## Contributing

Contributions are welcome! If you find bugs or have suggestions:

- Open an issue
- Submit a pull request
- Improve documentation
- Share your experiments

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

These tutorials are designed for educational purposes and draw inspiration from seminal papers in generative modeling:

- Ho et al. (2020) - Denoising Diffusion Probabilistic Models
- Lipman et al. (2023) - Flow Matching for Generative Modeling
- Song et al. (2021) - Score-Based Generative Modeling

---

Ready to start? Head to [Installation](installation.md) to set up your environment!

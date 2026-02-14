# Frequently Asked Questions

Find answers to common questions about the tutorials.

---

## General Questions

### What are these tutorials about?

These tutorials teach you how to implement modern generative models (diffusion models and flow matching) from scratch using PyTorch. You'll learn both the theory and practice through hands-on coding.

### Do I need a GPU?

No! All tutorials work on CPU. The 2D toy datasets train quickly even on CPU. A GPU helps with larger experiments but is not required.

### What's the difference between Tutorials 1 and 2?

- **Tutorial 1 (DDPM)**: Stochastic diffusion using SDEs
- **Tutorial 2 (Flow Matching)**: Deterministic diffusion using ODEs

Both generate data, but use different mathematical frameworks.

### How long does each tutorial take?

- **Tutorial 1**: 90-120 minutes
- **Tutorial 2**: 90-120 minutes  
- **Tutorial 3+**: 2-3 hours each (when released)

---

## Setup Questions

### Why use astral-uv instead of pip?

astral-uv is 10-100x faster than pip and provides deterministic dependency resolution. It's also compatible with the existing pip ecosystem, so you can switch back anytime.

### Can I use conda instead?

Yes! While the tutorials recommend astral-uv, you can use conda:

```bash
conda create -n tutorial python=3.11
conda activate tutorial
conda install pytorch numpy matplotlib jupyter
```

### Do I need both Jupyter Notebook and JupyterLab?

No, choose one:
- **Jupyter Notebook**: Classic, simpler
- **JupyterLab**: Modern, more features

Both work identically for these tutorials.

### How do I know which Python version to use?

Python 3.9 or higher. Recommended: Python 3.11.

```bash
python --version  # Should show 3.9+
```

---

## Running Tutorials

### Should I use Jupyter or CLI?

- **Jupyter**: Best for learning, experimentation, visualization
- **CLI**: Best for automation, batch processing, production

Most people start with Jupyter.

### Which kernel should I select in Jupyter?

Select "Tutorial Environment" or whichever kernel you registered:

```bash
python -m ipykernel install --user --name=tutorial-env
```

### Can I run tutorials in Google Colab?

Yes! Upload the notebook to Colab and install dependencies:

```python
!pip install torch numpy matplotlib
```

Then run the cells normally.

### How do I save my outputs?

Outputs are automatically saved to the `outputs/` directory in each tutorial folder.

---

## Tutorial Content

### What if I get stuck on a TODO?

1. Read the comments carefully
2. Review the mathematical explanation in the notebook
3. Check `flow_solutions.py` for reference (Tutorial 2)
4. Ask for help on GitHub issues

### Can I skip Tutorial 1?

Not recommended. Tutorial 2 builds on concepts from Tutorial 1. Start with DDPM to understand the basics.

### Are there video tutorials?

Not yet. These are self-paced text/code tutorials. Videos may be added in the future.

### Can I use these tutorials for a course?

Yes! These tutorials are open-source. Attribution appreciated but not required.

---

## Technical Questions

### What if my model isn't learning?

Common fixes:
- Check your learning rate (try 1e-3 or 1e-4)
- Verify data normalization
- Ensure loss is being computed correctly
- Train for more epochs
- Check your TODO implementations

### Why are my samples still noisy?

- Use more sampling steps (increase `n_euler_steps`)
- Train longer
- Check if loss converged
- Verify sampling procedure

### What's the difference between Euler and RK45?

- **Euler**: Simple, fixed step size, needs many steps
- **RK45**: Adaptive step size, fewer evaluations, more accurate

RK45 is usually better but harder to understand.

### Can I use these models for images?

The tutorials use 2D data for visualization. To scale to images:
- Modify model architecture (use U-Nets)
- Increase model capacity
- Train on image datasets (MNIST, CIFAR-10)
- Be prepared for longer training times

---

## Troubleshooting

### Module not found error

```bash
# Install package in editable mode
source .venv/bin/activate
uv pip install -e .
```

### Kernel not showing in Jupyter

```bash
# Reinstall kernel
python -m ipykernel install --user --name=tutorial-env
```

### CUDA out of memory

- Reduce batch size
- Use CPU instead
- Close other programs using GPU

### Import errors in notebook

Make sure you selected the correct kernel ("Tutorial Environment").

---

## Contributing

### How can I contribute?

- Report bugs via GitHub issues
- Submit pull requests for fixes
- Improve documentation
- Add new features
- Help others in discussions

### Can I add my own tutorial?

Yes! Follow the existing structure and submit a PR.

### How do I report a bug?

Open an issue on GitHub with:
- Error message
- Steps to reproduce
- System information
- Expected vs actual behavior

---

## Best Practices

### How should I structure my experiments?

```python
# 1. Create config dictionary
config = {
    "dataset": "moons",
    "learning_rate": 1e-3,
    ...
}

# 2. Run experiment
results = run_experiment(config)

# 3. Save outputs
save_results(results, "experiment_1")
```

### How do I compare different configurations?

Run multiple experiments with different configs and compare outputs:

```python
for lr in [1e-3, 1e-4, 1e-5]:
    config["learning_rate"] = lr
    results = run_experiment(config)
    save_results(results, f"lr_{lr}")
```

### Should I commit generated outputs?

Generally no. Add `outputs/` to `.gitignore`. Only commit code.

---

## Next Steps

### What should I do after completing the tutorials?

1. Read the original papers
2. Experiment with your own datasets
3. Try scaling to images
4. Explore advanced topics (guidance, conditioning)
5. Implement improvements (better architectures, faster sampling)

### Where can I learn more?

- Original papers (linked in each tutorial)
- [HuggingFace Diffusion Models Course](https://github.com/huggingface/diffusion-models-class)
- [Lilian Weng's Blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- Research papers on arXiv

### How do I stay updated?

- Watch the GitHub repository
- Check the documentation website
- Follow updates in README

---

Still have questions? Open an issue on [GitHub](https://github.com/yourusername/generative-tutorials/issues)!

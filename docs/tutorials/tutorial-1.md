# Tutorial 1: Polynomial Regression & Bias-Variance Tradeoff

Learn the foundations of machine learning by implementing polynomial regression from scratch and understanding the critical bias-variance tradeoff.

## Overview

This tutorial provides a hands-on introduction to **polynomial regression with PyTorch**, demonstrating one of the most fundamental concepts in machine learning: the **bias-variance tradeoff**.

**Duration:** 90-120 minutes

**Difficulty:** Beginner to Intermediate

---

## Learning Objectives

By the end of this tutorial, you will:

- Understand PyTorch tensor operations and automatic differentiation
- Learn to create custom model classes using `nn.Module`
- Implement forward passes and parameter optimization for arbitrary polynomial orders
- Grasp the training loop structure and parameter evolution
- **Visualize and understand the bias-variance tradeoff**
- Recognize underfitting (high bias) and overfitting (high variance)
- Learn when model complexity matches data complexity
- Master feature normalization for numerical stability with high-order polynomials

---

## Prerequisites

**Required Knowledge:**

- Python programming (basic to intermediate)
- Basic calculus (derivatives, chain rule)
- Linear algebra (vectors, matrices)
- Probability basics (Gaussian distributions, mean, variance)
- No prior PyTorch or ML experience required!

**Required Setup:**

- Complete [Installation Guide](../installation.md)
- Complete [Getting Started Guide](../getting-started.md)
- Python 3.8+ with PyTorch, matplotlib, numpy

---

## Tutorial Structure

The tutorial consists of:

1. **Jupyter Notebook** - Interactive, step-by-step learning (`tutorial_poly_fit.ipynb`)
2. **Python Scripts** - Modular, reusable code (`polynomial_tutorial/` package)
3. **Visualizations** - Plots showing parameter evolution, fitted curves, and bias-variance scenarios
4. **Documentation** - Comprehensive guides (README.md, this file)

---

## How to Run This Tutorial

You have **two options** for running this tutorial:

### Option 1: Jupyter Notebook (Recommended for Learning)

**Best for:** Interactive exploration, learning, experimentation

**Steps:**

1. Navigate to the tutorial directory:
   ```bash
   cd tutorial_1_polynomial_fit
   ```

2. Activate your virtual environment:
   ```bash
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\Activate.ps1  # Windows
   ```

3. Start Jupyter:
   ```bash
   jupyter lab
   # OR
   jupyter notebook
   ```

4. Open `tutorial_poly_fit.ipynb`

5. Run the cells sequentially

**Alternative: VSCode**

1. Open VSCode:
   ```bash
   code tutorial_1_polynomial_fit
   ```

2. Open `tutorial_poly_fit.ipynb`

3. Select your Python environment (the one with PyTorch installed)

4. Run cells using the play button or `Shift+Enter`

---

### Option 2: Terminal/CLI (For Automation)

**Best for:** Running complete experiments, batch processing, automated testing

**Steps:**

1. Navigate to the tutorial directory:
   ```bash
   cd tutorial_1_polynomial_fit
   ```

2. Activate your virtual environment:
   ```bash
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\Activate.ps1  # Windows
   ```

3. Run the main script:
   ```bash
   python -m polynomial_tutorial.main
   ```

   **OR:**
   
   ```bash
   python polynomial_tutorial/main.py
   ```

4. Check outputs:
   - Console: Training progress with logging
   - Plots: Two visualization windows will appear
   - Files: `parameter_evolution.png` and `polynomial_regression_results.png`

**Customize Configuration:**

Edit `polynomial_tutorial/main.py` to change:
- Data polynomial order (complexity of true function)
- Regressor polynomial order (complexity of model)
- Number of training samples
- Noise level (standard deviation)
- Training epochs and learning rate
- Feature normalization settings

---

## What's Inside

### Python Package: `polynomial_tutorial/`

```
polynomial_tutorial/
├── __init__.py          # Package initialization
├── main.py              # Entry point, experiment configuration
├── LinearRegressor.py   # PolynomialRegressor model class
├── train.py             # Training loop with parameter tracking
├── loss.py              # Loss functions (MSE, RMSE, MAE)
├── utils.py             # Feature normalization utilities
└── logger.py            # Professional logging configuration
```

**Key Files:**

- **`LinearRegressor.py`**: Polynomial regression model
  - `PolynomialRegressor`: Arbitrary nth-order polynomials
  - `LinearRegressor`: Special case (order=1) for backward compatibility
  - Learnable coefficients as `nn.Parameter`
  - Forward pass with polynomial feature computation

- **`train.py`**: Complete training pipeline
  - Training loop with gradient descent
  - Parameter history tracking (for visualization)
  - Model evaluation functions
  - Professional logging

- **`utils.py`**: Feature normalization toolkit
  - `FeatureNormalizer`: Prevents NaN for high-order polynomials
  - Three normalization methods (symmetric, standardization, min-max)
  - Fit, transform, inverse_transform operations
  - Educational demonstrations

- **`main.py`**: Experiment orchestration
  - Data generation with configurable noise
  - Automatic normalization for orders > 2
  - Training with parameter tracking
  - Comprehensive visualizations
  - Bias-variance scenario analysis

---

## Notebook Structure

The Jupyter notebook is organized into 12 parts:

### Part 1: Setup and Imports
- Library imports
- Logging configuration
- Environment verification

### Part 2: Configuration
- Experiment parameters
- Data/model polynomial orders
- Training hyperparameters
- Understanding the three bias-variance scenarios

### Part 3: Data Generation
- Creating polynomial data with noise
- Understanding the true function
- Visualizing generated data
- The irreducible error concept

### Part 4: Feature Normalization
- Why normalization is critical for high orders
- The numerical stability problem
- Applying normalization to prevent NaN
- Demonstrating the normalization effect

### Part 5: Model Definition
- The `PolynomialRegressor` class
- Learnable parameters
- Forward pass implementation
- Bias-variance scenario identification

### Part 6: Training
- Standard gradient descent loop
- Forward/backward passes
- Parameter updates
- Training progress logging

### Part 7: Evaluation
- Model performance metrics
- MSE and RMSE computation
- Interpreting results
- Comparing to noise level

### Part 8: Visualization - Parameter Evolution
- Tracking coefficient convergence
- Understanding which parameters matter
- Convergence speed analysis

### Part 9: Visualization - Fitted Curve
- Comparing predictions to true function
- Identifying underfitting/overfitting visually
- Loss convergence curves

### Part 10: Understanding the Results
- Interpreting the bias-variance scenario
- What the metrics mean
- When to be concerned

### Part 11: Experiments to Try
- Underfitting examples
- Overfitting examples
- Noise sensitivity analysis
- Sample size effects

### Part 12: Key Takeaways
- Summary of learned concepts
- Professional ML practices
- Next steps for learning

---

## Expected Outputs

When you run the tutorial, you'll generate:

**Console Output:**

- Structured logging with timestamps
- Training progress every 400 epochs
- Parameter values during training
- Final metrics and scenario analysis
- Pedagogical summaries and interpretations

**Visualizations:**

- `parameter_evolution.png` - How each coefficient converges over epochs
- `polynomial_regression_results.png` - Fitted curve vs true function + loss curve

**Example visualizations you'll create:**

**Parameter Evolution:**
Shows each coefficient's trajectory toward its optimal value
- Helps understand which terms are important
- Visualizes convergence dynamics
- Different parameters converge at different rates

**Fitted Curve vs True Function:**
- Left panel: Data points, fitted curve, true function
- Right panel: Training loss on log scale
- Clear visual indication of underfitting/overfitting
- Scenario labeled in title (High Bias / Balanced / High Variance)

---

## Key Concepts Covered

### 1. The Bias-Variance Tradeoff

The fundamental machine learning dilemma:

$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

**You'll learn:**
- **Bias:** Error from wrong model assumptions (too simple)
- **Variance:** Error from sensitivity to training data (too complex)
- **Irreducible Error:** Noise in the data (cannot be reduced)

**Three scenarios:**
- Model too simple → High Bias (underfitting)
- Model just right → Balanced (optimal)
- Model too complex → High Variance (overfitting)

### 2. Polynomial Regression

Fitting functions of the form:

$$y = a_0 + a_1 x + a_2 x^2 + \cdots + a_n x^n$$

**You'll implement:**
- Arbitrary polynomial orders
- Learnable coefficients as parameters
- Polynomial feature computation
- Efficient matrix multiplication approach

### 3. Feature Normalization (CRITICAL!)

When $x \in [0, 10]$, polynomial features explode:

| Feature | Range | Problem |
|---------|-------|---------|
| $x^3$ | [0, 1000] | ⚠️ Getting large |
| $x^4$ | [0, 10,000] | 🚨 Very large |
| $x^5$ | [0, 100,000] | 💥 **NaN values!** |

**Solution:** Normalize $x$ to $[-1, 1]$ **before** computing powers:
- All $(x_{\text{norm}})^n$ stay bounded
- Gradients remain stable
- Training succeeds for any polynomial order

**You'll master:**
- When normalization is necessary
- How to normalize properly
- Denormalizing for interpretable plots
- Professional ML practices

### 4. Training with PyTorch

Standard gradient descent optimization:

$$\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$$

**You'll learn:**
- Creating `nn.Module` classes
- Forward and backward passes
- Automatic differentiation (autograd)
- Parameter updates with optimizers
- Tracking training history

---

## Experiments to Try

After completing the tutorial, try these experiments:

### Easy (15-30 minutes each)

1. **Underfitting Demo:**
   ```python
   data_poly_order = 3          # Cubic data
   regressor_poly_order = 1     # Linear model (too simple!)
   ```
   **Observe:** High training loss, fitted line can't capture curvature

2. **Overfitting Demo:**
   ```python
   data_poly_order = 2          # Quadratic data
   regressor_poly_order = 6     # 6th order model (too complex!)
   ```
   **Observe:** Very low training loss, wiggly fit, extra coefficients ≈ 0

3. **Noise Sensitivity:**
   ```python
   noise_std = 0.1  # vs 0.5 vs 1.0 vs 2.0
   ```
   **Question:** How does noise affect overfitting?

### Medium (30-60 minutes each)

4. **Sample Size Effect:**
   ```python
   n_samples = 20  # vs 50 vs 100 vs 500
   ```
   **Question:** Does more data help with overfitting?

5. **Learning Rate Exploration:**
   ```python
   learning_rate = 0.001  # vs 0.01 vs 0.1
   ```
   **Question:** How does learning rate affect convergence?

6. **Compare Normalization Methods:**
   - Try 'symmetric' vs 'standardize' vs 'minmax'
   - Which works best for polynomials?

### Advanced (1-2 hours each)

7. **Implement Cross-Validation:**
   - Split data into train/validation/test
   - Use validation to select optimal polynomial order
   - Implement k-fold cross-validation

8. **Add L2 Regularization (Ridge Regression):**
   ```python
   loss = mse + lambda_reg * torch.sum(model.coeffs ** 2)
   ```
   - Prevents overfitting
   - Shrinks coefficients toward zero
   - Compare with/without regularization

9. **Automatic Model Selection:**
   - Implement AIC or BIC criteria
   - Automatically choose polynomial order
   - Compare to cross-validation approach

10. **Scale to Real Data:**
    - Load actual dataset (e.g., housing prices)
    - Apply polynomial regression
    - Compare different orders on real data

---

## Common Issues & Solutions

### Training loss becomes NaN

**Problem:** Loss shows `nan` after a few epochs

**Solutions:**
- The tutorial now **automatically applies normalization** for orders > 2
- If you disabled normalization: re-enable it
- Reduce learning rate (try 0.001)
- Check that data is properly normalized

**Why this happens:**
- Polynomial features like $x^5$ = 100,000 cause gradient explosion
- Normalization keeps all features in [-1, 1]

### Parameters don't match true values

**Problem:** Fitted coefficients don't equal true coefficients

**Expected behavior when using normalization:**
- Fitted coefficients are in **normalized scale**
- True coefficients are in **original scale**
- Cannot directly compare (different coordinate systems)
- Instead, compare predictions visually in plots

**Solution:**
- Check the fitted curve plot — does it match visually?
- Compare RMSE to noise level σ
- Focus on prediction quality, not coefficient values

### Model underfits even with correct order

**Problem:** High training loss despite matching orders

**Solutions:**
- Train longer (increase `num_epochs`)
- Increase learning rate slightly
- Check for bugs in data generation
- Verify noise isn't too high

### Generated plots don't show

**Problem:** Training completes but no plots appear

**Solutions:**
- In Jupyter: Use `%matplotlib inline`
- In terminal: Check if matplotlib backend supports display
- Manually save plots: `plt.savefig('output.png')`
- Check for errors in plotting functions

---

## Theoretical Background

For deeper understanding, refer to these resources:

**Core Textbooks:**

- **The Elements of Statistical Learning** (Hastie, Tibshirani, Friedman)
  - Chapter 2: Overview of Supervised Learning
  - Chapter 3: Linear Methods for Regression
  - Chapter 7: Model Assessment and Selection

- **Pattern Recognition and Machine Learning** (Bishop)
  - Chapter 1: Introduction (bias-variance tradeoff)
  - Chapter 3: Linear Models for Regression
  - Polynomial curve fitting example

- **Deep Learning** (Goodfellow, Bengio, Courville)
  - Chapter 5: Machine Learning Basics
  - Section 5.4: Capacity, Overfitting, and Underfitting

**Fundamental Papers:**

- Geman et al. (1992): "Neural Networks and the Bias/Variance Dilemma"
- Domingos (2000): "A Unified Bias-Variance Decomposition"

**Online Resources:**

- [Bias-Variance Interactive Visualization](http://www.r2d3.us/visual-intro-to-machine-learning-part-2/)
- [scikit-learn: Underfitting vs Overfitting](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html)
- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)

---

## Next Steps

After completing Tutorial 1:

- **Tutorial 2:** [Perceptron to Deep Neural Networks](tutorial-2.md) - Build multi-layer networks
- **Tutorial 3:** [DNNs to Transformers](tutorial-3.md) - Modern architectures
- Experiment with different polynomial orders and noise levels
- Read Chapter 1 of "Pattern Recognition and Machine Learning" (Bishop)
- Explore regularization techniques (L1, L2, elastic net)
- Try polynomial regression on real datasets

---

## Quick Reference

**Start Jupyter:**
```bash
cd tutorial_1_polynomial_fit
source .venv/bin/activate  # or appropriate activation
jupyter lab
```

**Run CLI:**
```bash
cd tutorial_1_polynomial_fit
source .venv/bin/activate
python -m polynomial_tutorial.main
```

**Common imports:**
```python
from polynomial_tutorial.LinearRegressor import PolynomialRegressor
from polynomial_tutorial.train import train_model
from polynomial_tutorial.utils import FeatureNormalizer
```

**Quick experiment:**
```python
# Underfitting
data_poly_order = 3
regressor_poly_order = 1

# Overfitting
data_poly_order = 2
regressor_poly_order = 6

# Good fit
data_poly_order = 3
regressor_poly_order = 3
```

---

## Additional Resources

### Included Scripts

**Run the comprehensive demo:**
```bash
python polynomial_tutorial/bias_variance_demo.py
```

This runs all three scenarios (underfitting, good fit, overfitting) side-by-side and creates a 3×3 comparison plot!

**Demonstrate normalization effect:**
```bash
python -c "from polynomial_tutorial.utils import demonstrate_normalization_effect; demonstrate_normalization_effect()"
```

Shows feature ranges with/without normalization — great for understanding why it's necessary.

### Files Included

- `QUICKSTART.md` - Quick start guide for students
- `BUGFIX_SUMMARY.md` - Technical notes on bug fixes
- `NORMALIZATION_SOLUTION.md` - Deep dive into feature normalization
- `README.md` - Comprehensive project documentation

---

## Need Help?

**Common Questions:**

- **Q:** Why use feature normalization?
  - **A:** Prevents NaN values with high-order polynomials (orders > 2)

- **Q:** When should I use which polynomial order?
  - **A:** That's the point of the tutorial! Match model complexity to data complexity

- **Q:** Can I compare fitted coefficients to true coefficients?
  - **A:** Only if NOT using normalization. With normalization, compare predictions instead.

- **Q:** What's the best learning rate?
  - **A:** Default 0.01 works well with normalization. Experiment between 0.001 and 0.1.

**Troubleshooting:**

- Check [FAQ](../faq.md)
- See [Troubleshooting Guide](../troubleshooting.md)
- Review inline code documentation
- Check the comprehensive README.md

**Support:**

- Open an issue on GitHub
- Ask in discussion forums
- Consult your instructor

---

!!! success "Ready to Start"
    Head to `tutorial_1_polynomial_fit/tutorial_poly_fit.ipynb` and begin learning!
    
    Or run `python -m polynomial_tutorial.main` to see it in action immediately!

---

## Summary

This tutorial teaches:

✅ Polynomial regression from scratch with PyTorch  
✅ The fundamental bias-variance tradeoff  
✅ Feature normalization for numerical stability  
✅ Professional ML coding practices  
✅ How to visualize and interpret model behavior  
✅ When simple is too simple and complex is too complex  

**Key Insight:** The goal is NOT to minimize training loss, but to build models that **generalize to new data**!

**Remember:** Low training loss can indicate overfitting. The sweet spot is where bias and variance are balanced.

Happy learning! 🚀

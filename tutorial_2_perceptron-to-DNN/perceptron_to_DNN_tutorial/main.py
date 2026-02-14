"""
main_regularization.py
=======================
Tutorial: Vanishing Gradients, Regularization & Overfitting Detection

This script demonstrates THREE critical concepts in deep learning:

1. VANISHING GRADIENT PROBLEM:
   Deep networks are needed for complex functions (high-order polynomials),
   but gradients can vanish with certain activation functions (tanh, sigmoid).
   ReLU solves this problem!

2. OVERFITTING DETECTION:
   Use train/validation/test splits to detect overfitting.
   Compare training loss vs. validation loss to see if model generalizes.
   Large gap = overfitting (memorizing training data).

3. REGULARIZATION:
   High-order polynomials create unstable/spikey loss curves during training.
   Regularization (L1/L2, dropout) stabilizes training and prevents overfitting.
   Keeps validation loss close to training loss!

Key Pedagogical Insights:
   - Depth is power (can learn complex functions)
   - Depth is dangerous (vanishing gradients)
   - ReLU is the solution (no vanishing)
   - Train/val/test splits detect overfitting
   - Regularization prevents overfitting
   - Monitor validation loss during training

Author: ML Tutorial Series
Target Audience: PhD students and early career postdocs

Usage:
    python main_regularization.py
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from MultiLayerPerceptron import MultiLayerPerceptron
from train import train_model_with_gradient_tracking, train_model_with_validation_tracking
from utils import FeatureNormalizer
from plotting import (plot_gradient_flow, plot_layer_gradient_norms, 
                       plot_regularization_comparison, plot_results)

# Logging
import logging
from logger import configure_logging
configure_logging()
logger = logging.getLogger(__name__)


def generate_high_order_polynomial_data(coeffs_true,
                                        poly_order,
                                        n_samples=100,
                                        x_min=0, x_max=10,
                                        noise_std=0.5,
                                        seed=None):
    """
    Generate data from a high-order polynomial.
    
    High-order polynomials (order > 5) are challenging:
    - Require deep/wide networks to learn
    - Create unstable loss landscapes
    - Prone to overfitting
    
    This makes them perfect for demonstrating:
    1. Why we need deep networks
    2. Vanishing gradient problem
    3. Benefits of regularization
    4. Train/validation/test splits for detecting overfitting
    
    Args:
        coeffs_true (list): Polynomial coefficients [c0, c1, ..., cn]
        poly_order (int): Order of polynomial (should match len(coeffs_true)-1)
        n_samples (int): Number of data points to generate
        x_min, x_max (float): Range for input x
        noise_std (float): Standard deviation of Gaussian noise
        seed (int, optional): Random seed for reproducibility
    
    Returns:
        tuple: (x, y_noisy) tensors
    """
    assert len(coeffs_true) == poly_order + 1
    
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    x = torch.linspace(x_min, x_max, n_samples)
    
    y_true = torch.zeros(n_samples)
    for i, coeff in enumerate(coeffs_true):
        y_true += coeff * (x ** i)
    
    noise = torch.randn(n_samples) * noise_std
    y_noisy = y_true + noise
    
    return x, y_noisy


def main():
    """
    Main function demonstrating vanishing gradients and regularization.
    """
    # ========================================================================
    # INTRODUCTION
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info(" TUTORIAL: VANISHING GRADIENTS, OVERFITTING & REGULARIZATION")
    logger.info("="*70)
    logger.info("""
    This tutorial demonstrates THREE critical concepts:
    
    PART 1: VANISHING GRADIENT PROBLEM
    ------------------------------------
    Deep networks are powerful but suffer from vanishing gradients
    with certain activation functions (tanh, sigmoid).
    
    Why it happens:
    - Backpropagation uses chain rule: ∂L/∂W₁ = ∂L/∂h₃ · ∂h₃/∂h₂ · ∂h₂/∂h₁ · ∂h₁/∂W₁
    - If each term < 1, product shrinks exponentially with depth
    - tanh'(x) ∈ (0, 1], sigmoid'(x) ∈ (0, 0.25] → gradients vanish!
    - ReLU'(x) = 1 for x > 0 → no vanishing!
    
    PART 2: OVERFITTING DETECTION
    -------------------------------
    Split data into train/validation/test to detect overfitting:
    - Training loss always decreases with more epochs
    - But does the model generalize to unseen data?
    - Compare train vs. validation loss curves
    - Large gap = overfitting (memorizing training data)
    - Small gap = good generalization
    
    PART 3: REGULARIZATION
    -----------------------
    High-order polynomials create unstable training AND overfitting:
    - Loss curves have spikes and oscillations
    - Model overfits to training noise
    - Training is unpredictable
    
    Solutions:
    - L1 regularization: Promotes sparse weights
    - L2 regularization: Keeps weights small
    - Dropout: Randomly drops neurons during training
    - All keep validation loss close to training loss!
    
    Let's see these effects in action!
    """)
    logger.info("="*70)
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    logger.info("\n[CONFIGURATION]")
    
    # Data: High-order polynomial (challenging to learn!)
    poly_order = 9  # 7th order polynomial
    #coeffs_true = coeffs_true = [1.0, 0.5, -0.03, 0.015, -0.001, -0.0003, 0.00005, -0.000005, -0.0000001, 0.00000002]
    #coeffs_true = coeffs_true = [10.0, 0.5, -0.033, 0.015, -0.001, -0.0003, 0.00005, -0.000005, -0.0000001, 0.00000002]
    coeffs_true = coeffs_true = [10.0, 0.5, -0.04, 0.015, -0.001, -0.0003, 0.000055, -0.000005, -0.0000001, 0.00000002] 
    n_samples_train = 200
    n_samples_valid = 200
    n_samples_test = 200
    noise_std = 2.5
    
    # Training
    num_epochs = 10000
    learning_rate = 0.005
    
    # Network architecture (DEEP network to demonstrate vanishing gradient)
    #architecture_deep = [1, 32, 32, 32, 32, 32, 32, 32, 32, 32, 1]  # 9 hidden layers
    architecture_deep = [1, 128, 128, 128, 128, 128, 128, 128, 128, 128, 1]  # 9 hidden layers
    
    
    # Regularization settings
    lambda_l2 = 0.01  # L2 regularization strength
    lambda_l1 = 0.002  # L1 regularization strength
    dropout_rate = 0.2  # Dropout probability
    
    logger.info(f"Polynomial order: {poly_order} (high-order!)")
    logger.info(f"True coefficients: {coeffs_true}")
    logger.info(f"Training samples: {n_samples_train}")
    logger.info(f"Validation samples: {n_samples_valid}")
    logger.info(f"Test samples: {n_samples_test}")
    logger.info(f"Noise: {noise_std}")
    logger.info(f"Network architecture: {' → '.join(map(str, architecture_deep))}")
    logger.info(f"Training: {num_epochs} epochs, lr={learning_rate}")
    
    # ========================================================================
    # PART 1: VANISHING GRADIENT DEMONSTRATION
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info(" PART 1: VANISHING GRADIENT PROBLEM")
    logger.info("="*70)
    
    # Generate train/validation/test datasets
    logger.info("\n[STEP 1.1] Generating train/validation/test datasets...")
    
    # Training data
    x_t, y_t = generate_high_order_polynomial_data(
        coeffs_true=coeffs_true,
        poly_order=poly_order,
        n_samples=n_samples_train,
        x_min=0,
        x_max=10,
        noise_std=noise_std,
        seed=42
    )
    
    # Validation data (different noise, same distribution)
    x_valid, y_valid = generate_high_order_polynomial_data(
        coeffs_true=coeffs_true,
        poly_order=poly_order,
        n_samples=n_samples_valid,
        x_min=0,
        x_max=10,
        noise_std=noise_std,
        seed=123
    )
    
    # Test data (held out for final evaluation)
    x_test, y_test = generate_high_order_polynomial_data(
        coeffs_true=coeffs_true,
        poly_order=poly_order,
        n_samples=n_samples_test,
        x_min=0,
        x_max=10,
        noise_std=noise_std,
        seed=456
    )
    
    logger.info(f"✓ Training: {n_samples_train} samples, y range [{y_t.min():.2f}, {y_t.max():.2f}]")
    logger.info(f"✓ Validation: {n_samples_valid} samples, y range [{y_valid.min():.2f}, {y_valid.max():.2f}]")
    logger.info(f"✓ Test: {n_samples_test} samples, y range [{y_test.min():.2f}, {y_test.max():.2f}]")
    
    # Normalize features
    logger.info("\n[STEP 1.2] Normalizing features...")
    normalizer = FeatureNormalizer(method='symmetric')
    x_t = normalizer.fit_transform(x_t)
    x_valid = normalizer.transform(x_valid)  # Use train statistics!
    x_test = normalizer.transform(x_test)    # Use train statistics!
    logger.info("✓ Features normalized to [-1, 1] using training statistics")
    logger.info("  (Validation and test use same normalization as training)")
    
    # Test different activation functions
    logger.info("\n[STEP 1.3] Training deep networks with different activations...")
    logger.info("This will demonstrate the vanishing gradient problem!")
    
    activations_to_test = ['sigmoid', 'tanh', 'relu']
    gradient_histories = {}
    
    for activation in activations_to_test:
        logger.info(f"\n--- Training with {activation.upper()} activation ---")
        
        # Create deep network
        model = MultiLayerPerceptron(
            layer_sizes=architecture_deep,
            activation=activation,
            use_activation_output=False,
            dropout_rate=0.0  # No dropout for gradient analysis
        )
        
        # Train with gradient tracking (and validation monitoring)
        trained_model, history = train_model_with_gradient_tracking(
            model=model,
            x_train=x_t,
            y_train=y_t,
            x_valid=x_valid,
            y_valid=y_valid,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            reg_type='none',  # No regularization for pure gradient analysis
            print_every=500,
            verbose=True,
            track_gradients=True,
            track_per_sample_gradients=True  # Enable per-sample gradient distributions
        )
        
        # Store full history (needed for per-sample gradient distributions)
        gradient_histories[activation] = history
        
        # Debug: Check what's in the history
        logger.info(f"\nHistory keys for {activation}: {list(history.keys())}")
        if 'per_sample_gradient_distributions' in history:
            logger.info(f"  ✓ Per-sample gradients present: {len(history['per_sample_gradient_distributions'])} epochs")
            logger.info(f"  Epochs: {list(history['per_sample_gradient_distributions'].keys())}")
        else:
            logger.warning(f"  ✗ Per-sample gradients NOT in history!")
        
        # Analyze final gradient magnitudes
        final_grads = history['gradient_norms'][-1]
        layer_names = [name for name in final_grads.keys() if 'weight' in name]
        
        logger.info(f"\nFinal gradient magnitudes ({activation}):")
        for layer_name in layer_names:
            logger.info(f"  {layer_name}: {final_grads[layer_name]:.6e}")
        
        # Check for vanishing
        first_layer_grad = final_grads[layer_names[0]]
        last_layer_grad = final_grads[layer_names[-1]]
        if first_layer_grad > 0:
            ratio = last_layer_grad / first_layer_grad
            logger.info(f"\nGradient ratio (first/last layer): {ratio:.6e}")
            if ratio < 0.01:
                logger.info("⚠️  WARNING: VANISHING GRADIENT DETECTED!")
                logger.info("    Earlier layers receive tiny gradients and learn slowly.")
            else:
                logger.info("✓ Gradients flowing reasonably through all layers")
    
    # Visualize gradient flow
    logger.info("\n[STEP 1.4] Visualizing gradient flow...")
    example_model = MultiLayerPerceptron(architecture_deep, activation='relu')
    plot_gradient_flow(gradient_histories, example_model, activations_to_test,
                      save_name='vanishing_gradient_demo.png')
    
    # Visualize gradient distributions at different epochs
    logger.info("\n[STEP 1.5] Visualizing gradient norm distributions across epochs...")
    logger.info("Creating 3 separate plots: first epoch, middle epoch, last epoch")
    plot_layer_gradient_norms(gradient_histories, example_model, activations_to_test,
                             save_name='gradient_distributions')
    
    # Visualize best model fit (ReLU)
    logger.info("\n[STEP 1.6] Visualizing ReLU model fit (best for deep networks)...")
    logger.info("Training ReLU model with validation tracking for overfitting analysis...")
    # Get the ReLU model and history from the loop
    relu_model = MultiLayerPerceptron(architecture_deep, activation='relu', dropout_rate=0.0)
    relu_model, relu_history = train_model_with_gradient_tracking(
        model=relu_model,
        x_train=x_t,
        y_train=y_t,
        x_valid=x_valid,
        y_valid=y_valid,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        reg_type='none',
        print_every=num_epochs + 1,  # Silent
        verbose=False,
        track_gradients=False  # Don't need gradients for plotting
    )

    plot_results(
        x_t=x_t,
        y_t=y_t,
        model=relu_model,
        training_history=relu_history,
        coeffs_true=coeffs_true,
        data_poly_order=poly_order,
        model_name="Deep MLP with ReLU",
        normalizer=normalizer
        # show_validation=True is default, will show validation curves
    )

    
    
    logger.info("\n" + "="*70)
    logger.info(" KEY OBSERVATIONS - Vanishing Gradient:")
    logger.info("="*70)
    logger.info("""
    - SIGMOID: Severe vanishing! Gradients ~ 10⁻⁶ in early layers
    - TANH: Moderate vanishing, gradients ~ 10⁻³ in early layers
    - RELU: NO vanishing! Gradients stable across all layers
    
    CONCLUSION: Use ReLU for deep networks to avoid vanishing gradients!
    """)
    logger.info("="*70)
    
    # ========================================================================
    # PART 2: REGULARIZATION DEMONSTRATION
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info(" PART 2: REGULARIZATION FOR STABLE TRAINING")
    logger.info("="*70)
    
    logger.info("""
    High-order polynomials create unstable loss landscapes.
    Without regularization, training has:
    - Spikey loss curves
    - Oscillations
    - Potential divergence
    
    We'll compare:
    1. No regularization (baseline)
    2. L2 regularization
    3. L1 regularization
    4. Dropout regularization
    """)
    
    logger.info("\n[STEP 2.1] Training WITHOUT regularization...")
    logger.info("This will show overfitting: train loss decreases but validation loss increases!")
    
    model_no_reg = MultiLayerPerceptron(
        layer_sizes=architecture_deep,
        activation='relu',  # Use ReLU to avoid vanishing gradient
        dropout_rate=0.0
    )
    
    _, history_no_reg = train_model_with_validation_tracking(
        model=model_no_reg,
        x_train=x_t,
        y_train=y_t,
        x_valid=x_valid,
        y_valid=y_valid,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        reg_type='none',
        print_every=500,
        verbose=True,
        track_gradients=False  # Speed up training
    )
    
    logger.info("\n[STEP 2.2] Training WITH L2 regularization...")
    logger.info("L2 should prevent overfitting: validation loss stays close to training loss.")
    
    model_l2 = MultiLayerPerceptron(
        layer_sizes=architecture_deep,
        activation='relu',
        dropout_rate=0.0
    )
    
    _, history_l2 = train_model_with_validation_tracking(
        model=model_l2,
        x_train=x_t,
        y_train=y_t,
        x_valid=x_valid,
        y_valid=y_valid,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        reg_type='l2',
        lambda_l2=lambda_l2,
        print_every=500,
        verbose=True,
        track_gradients=False
    )
    
    logger.info("\n[STEP 2.3] Training WITH L1 regularization...")
    logger.info("L1 should also prevent overfitting with sparse weights.")
    
    model_l1 = MultiLayerPerceptron(
        layer_sizes=architecture_deep,
        activation='relu',
        dropout_rate=0.0
    )
    
    _, history_l1 = train_model_with_validation_tracking(
        model=model_l1,
        x_train=x_t,
        y_train=y_t,
        x_valid=x_valid,
        y_valid=y_valid,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        reg_type='l1',
        lambda_l1=lambda_l1,
        print_every=500,
        verbose=True,
        track_gradients=False
    )
    
    logger.info("\n[STEP 2.4] Training WITH dropout...")
    logger.info("Dropout should provide best overfitting prevention!")
    
    model_dropout = MultiLayerPerceptron(
        layer_sizes=architecture_deep,
        activation='relu',
        dropout_rate=dropout_rate
    )
    
    _, history_dropout = train_model_with_validation_tracking(
        model=model_dropout,
        x_train=x_t,
        y_train=y_t,
        x_valid=x_valid,
        y_valid=y_valid,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        reg_type='none',  # Dropout is the regularization
        print_every=500,
        verbose=True,
        track_gradients=False
    )
    
    # Compare results
    logger.info("\n[STEP 2.5] Comparing regularization approaches...")
    
    histories = [history_no_reg, history_l2, history_l1, history_dropout]
    model_names = ['No Regularization', f'L2 (λ={lambda_l2})', 
                   f'L1 (λ={lambda_l1})', f'Dropout (p={dropout_rate})']
    
    plot_regularization_comparison(histories, model_names,
                                  save_name='regularization_comparison.png')
    
    # Visualize best regularized model (Dropout)
    logger.info("\n[STEP 2.6] Visualizing best regularized model (Dropout)...")
    plot_results(
        x_t=x_test,
        y_t=y_test,
        model=model_dropout,
        training_history=history_dropout,  # Contains train + valid losses
        coeffs_true=coeffs_true,
        data_poly_order=poly_order,
        model_name=f"Deep MLP with Dropout (p={dropout_rate})",
        normalizer=normalizer
        # show_validation=True is default, will show train vs. validation curves
    )
    
    plot_results(
        x_t=x_test,
        y_t=y_test,
        model=model_l2,
        training_history=history_l2,  # Contains train + valid losses
        coeffs_true=coeffs_true,
        data_poly_order=poly_order,
        model_name=f"Deep MLP with L2 (l2={lambda_l2})",
        normalizer=normalizer
        # show_validation=True is default, will show train vs. validation curves
    )

    plot_results(
        x_t=x_test,
        y_t=y_test,
        model=model_l1,
        training_history=history_l1,  # Contains train + valid losses
        coeffs_true=coeffs_true,
        data_poly_order=poly_order,
        model_name=f"Deep MLP with L1 (l1={lambda_l1})",
        normalizer=normalizer
        # show_validation=True is default, will show train vs. validation curves
    )

    plot_results(
        x_t=x_test,
        y_t=y_test,
        model=model_no_reg,
        training_history=history_no_reg,  # Contains train + valid losses
        coeffs_true=coeffs_true,
        data_poly_order=poly_order,
        model_name=f"Deep MLP without no-reg",
        normalizer=normalizer
        # show_validation=True is default, will show train vs. validation curves
    )
    
    # Compute stability metrics
    logger.info("\nTraining stability and overfitting metrics:")
    for name, history in zip(model_names, histories):
        train_loss = np.array(history['train_loss'])
        valid_loss = np.array(history['valid_loss'])
        
        # Compute metrics
        final_train = train_loss[-1]
        final_valid = valid_loss[-1]
        overfitting_gap = final_valid - final_train
        train_std = np.std(train_loss[-500:])  # Std of last 500 epochs
        n_spikes = np.sum(np.abs(np.diff(train_loss)) > 0.1 * np.mean(train_loss))
        
        logger.info(f"\n{name}:")
        logger.info(f"  Final train loss: {final_train:.6f}")
        logger.info(f"  Final valid loss: {final_valid:.6f}")
        logger.info(f"  Overfitting gap: {overfitting_gap:+.6f}")
        logger.info(f"  Train loss std (last 500): {train_std:.6f}")
        logger.info(f"  Number of spikes: {n_spikes}")
        
        # Diagnose overfitting
        if overfitting_gap < 0.1:
            logger.info(f"  Status: ✓ Excellent generalization!")
        elif overfitting_gap < 0.5:
            logger.info(f"  Status: ⚠ Slight overfitting")
        else:
            logger.info(f"  Status: ⚠️ Significant overfitting!")
    
    logger.info("\n" + "="*70)
    logger.info(" KEY OBSERVATIONS - Regularization & Overfitting:")
    logger.info("="*70)
    logger.info("""
    - NO REGULARIZATION: Train loss low, but validation loss high → OVERFITTING!
    - L2: Validation loss stays closer to training loss → Less overfitting
    - L1: Similar to L2, with sparse weights
    - DROPOUT: Best generalization! Minimal overfitting gap
    
    CONCLUSION: Regularization prevents overfitting by keeping
    validation loss close to training loss!
    """)
    logger.info("="*70)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info(" TUTORIAL COMPLETED!")
    logger.info("="*70)
    logger.info("""
    YOU LEARNED:
    
    1. VANISHING GRADIENT PROBLEM:
       ✓ Deep networks needed for complex functions
       ✓ Sigmoid/tanh cause vanishing gradients
       ✓ ReLU solves the problem
       ✓ Visualized gradient flow through layers
    
    2. TRAIN/VALIDATION/TEST SPLITS:
       ✓ Training set: Learn parameters
       ✓ Validation set: Detect overfitting during training
       ✓ Test set: Final evaluation (not used during training!)
       ✓ Always normalize using training statistics
    
    3. OVERFITTING DETECTION:
       ✓ Compare training loss vs. validation loss
       ✓ Small gap → Good generalization
       ✓ Large gap → Overfitting (memorizing training data)
       ✓ Visualized with train/validation curves
    
    4. REGULARIZATION:
       ✓ Prevents overfitting by penalizing complexity
       ✓ L1: Promotes sparsity (some weights → 0)
       ✓ L2: Keeps all weights small
       ✓ Dropout: Most robust, prevents overfitting best
       ✓ Keeps validation loss close to training loss!
    
    PRACTICAL RECOMMENDATIONS:
    1. Always split data: train/validation/test
    2. Monitor validation loss during training
    3. Use ReLU activation for deep networks
    4. Add regularization (L2 or dropout) for complex models
    5. Stop training when validation loss starts increasing
    6. Evaluate final model on held-out test set
    """)
    logger.info("="*70)
    
    # ========================================================================
    # EXERCISES
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info(" EXERCISES FOR STUDENTS")
    logger.info("="*70)
    logger.info("""
    EXPERIMENT 1: Dataset Size Effect
    - Vary n_samples_train: 50, 100, 200, 500
    - Keep validation/test fixed
    - Question: How does training size affect overfitting gap?
    
    EXPERIMENT 2: Validation Set Size
    - Keep training fixed at 100 samples
    - Vary n_samples_valid: 20, 50, 100, 200
    - Question: Is validation loss more reliable with more samples?
    
    EXPERIMENT 3: Early Stopping
    - Modify code to stop when validation loss increases for 10 epochs
    - Compare final test performance with/without early stopping
    - Question: Does early stopping improve generalization?
    
    EXPERIMENT 4: Regularization Strength
    - For no-reg model: Observe large train/valid gap
    - Sweep lambda_l2: 0.0001, 0.001, 0.01, 0.1
    - Question: What lambda gives smallest validation loss?
    
    EXPERIMENT 5: Combined Regularization
    - Try L2 + dropout together
    - Compare to L2 alone and dropout alone
    - Question: Is combination better than individual?
    
    EXPERIMENT 6: Test Set Evaluation
    - After training all models, evaluate on test set
    - Compute test loss for each model
    - Question: Does lowest validation loss → lowest test loss?
    
    CHALLENGE: Implement Early Stopping
    - Stop training when validation loss increases for N consecutive epochs
    - Save best model (lowest validation loss)
    - Evaluate on test set
    - Expected: Better generalization than training to convergence
    """)
    logger.info("="*70 + "\n")
    logger.info("""
       ✓ L2: Keeps all weights small
       ✓ Dropout: Most robust, prevents overfitting
       ✓ All stabilize loss curves
    
    PRACTICAL RECOMMENDATIONS:
    1. Use ReLU activation for deep networks
    2. Add regularization for complex problems
    3. Start with L2 or dropout
    4. Monitor gradient magnitudes during training
    5. If loss is spikey, increase regularization
    """)
    logger.info("="*70)
    
    # ========================================================================
    # EXERCISES
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info(" EXERCISES FOR STUDENTS")
    logger.info("="*70)
    logger.info("""
    EXPERIMENT 1: Network Depth
    - Try architectures: [1, 32, 1], [1, 32, 32, 1], [1, 32, 32, 32, 32, 1]
    - How does vanishing gradient change with depth?
    - At what depth does it become severe for tanh/sigmoid?
    
    EXPERIMENT 2: Activation Functions
    - Try leaky ReLU: ReLU with small negative slope
    - Try ELU (Exponential Linear Unit)
    - Compare gradient flow patterns
    
    EXPERIMENT 3: Regularization Strength
    - Vary lambda_l2: 0.0001, 0.001, 0.01, 0.1, 1.0
    - How does it affect training stability?
    - Is there an optimal value?
    
    EXPERIMENT 4: Dropout Rate
    - Try dropout_rate: 0.1, 0.3, 0.5, 0.7
    - How does it affect final accuracy?
    - What's the best rate for this problem?
    
    EXPERIMENT 5: Polynomial Order
    - Try orders: 3, 5, 7, 9, 11
    - At what order does training become unstable?
    - Does regularization help more for higher orders?
    
    EXPERIMENT 6: Combine Regularizations
    - Use L2 + dropout together
    - Use L1 + L2 (elastic net)
    - Which combination works best?
    
    CHALLENGE: Gradient Clipping
    - Implement gradient clipping: clip gradients to max norm
    - Does this help with training stability?
    - Compare to regularization approaches
    """)
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()

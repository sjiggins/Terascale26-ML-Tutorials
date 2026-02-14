"""
plotting.py
===========
Plotting utilities for neural network training visualization.

Contains functions for:
- Gradient flow visualization
- Regularization comparison
- Training results visualization
- Gradient norm distribution analysis

Author: ML Tutorial Series
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from MultiLayerPerceptron import MultiLayerPerceptron

# Logging
import logging
logger = logging.getLogger(__name__)


def plot_gradient_flow(gradient_norms_history, model, activations_tested, save_name='gradient_flow.png'):
    """
    Visualize gradient flow through the network.
    
    Creates plots showing:
    1. Gradient magnitude per layer over training
    2. Comparison across different activation functions
    3. Evidence of vanishing/exploding gradients
    
    Args:
        gradient_norms_history: Dict mapping activation -> gradient history
                               Can be either:
                               - List of gradient dicts (legacy format)
                               - Full training history dict with 'gradient_norms' key
        model: Example model for layer names
        activations_tested: List of activation functions tested
        save_name: Output filename
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Vanishing Gradient Problem: Effect of Activation Functions', 
                fontsize=16, fontweight='bold')
    
    # Get layer names
    layer_names = [name for name, _ in model.named_parameters() if 'weight' in name]
    n_layers = len(layer_names)
    
    # Color map for layers (earlier layers in warmer colors)
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, n_layers))
    
    # Plot 1: Gradient norms over time for each activation
    for idx, activation in enumerate(activations_tested):
        ax = axes[idx // 2, idx % 2]
        
        # Handle both formats: full history dict or just gradient_norms list
        history = gradient_norms_history[activation]
        if isinstance(history, dict) and 'gradient_norms' in history:
            grad_history = history['gradient_norms']
        else:
            grad_history = history
        
        epochs = range(len(grad_history))
        
        # Plot each layer's gradient norm
        for layer_idx, layer_name in enumerate(layer_names):
            grad_norms = [grad_dict.get(layer_name, 0) for grad_dict in grad_history]
            
            # Skip if all zeros
            if max(grad_norms) == 0:
                continue
            
            layer_display_name = f"Layer {layer_idx + 1}"
            ax.plot(epochs, grad_norms, label=layer_display_name, 
                   color=colors[layer_idx], linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Gradient Norm (L2)', fontsize=12)
        ax.set_title(f'Activation: {activation.upper()}', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, which='both')
        
        # Add annotation for vanishing gradient
        final_grad_first_layer = [grad_dict.get(layer_names[0], 0) for grad_dict in grad_history][-1]
        final_grad_last_layer = [grad_dict.get(layer_names[-1], 0) for grad_dict in grad_history][-1]
        
        if final_grad_first_layer > 0 and final_grad_last_layer > 0:
            ratio = final_grad_first_layer / final_grad_last_layer
            if ratio < 0.1:
                ax.text(0.05, 0.05, f'⚠️ VANISHING!\nLayer 1/Layer {n_layers} = {ratio:.2e}',
                       transform=ax.transAxes, fontsize=10, 
                       bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
            elif ratio > 0.9:
                ax.text(0.05, 0.05, f'✓ STABLE\nLayer 1/Layer {n_layers} = {ratio:.2f}',
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    logger.info(f"Gradient flow visualization saved as '{save_name}'")
    plt.show()


def plot_layer_gradient_norms(gradient_norms_history, model, activations_tested, 
                               save_name='gradient_distributions'):
    """
    Visualize distribution of signed gradient values across all training samples.
    
    Creates THREE SEPARATE plots showing:
    - Histogram of signed gradient values at FIRST epoch
    - Histogram of signed gradient values at MIDDLE epoch  
    - Histogram of signed gradient values at LAST epoch
    
    Each plot shows a grid:
    - Rows: Different activation functions (sigmoid, tanh, relu)
    - Columns: Different layers
    - Content: 1D histogram showing distribution of ALL gradient values (with signs)
              across all samples for that layer
    
    This visualization reveals:
    - Full distribution shape (Gaussian, bimodal, skewed?)
    - Balance between positive and negative gradients
    - Are gradients vanishing (distribution concentrated near zero)?
    - Are gradients exploding (wide distribution to large values)?
    - Different behavior across activation functions
    
    Args:
        gradient_norms_history: Dict mapping activation -> training_history
                               training_history must contain 'per_sample_gradient_distributions'
                               Format: {epoch_idx: {layer_name: [array1, array2, ...]}}
                               where each array contains signed gradient values for one sample
        model: Example model for layer names
        activations_tested: List of activation functions tested
        save_name: Base name for output files (will append epoch identifier)
    
    Pedagogical Note:
        This plot shows the DISTRIBUTION of signed gradients across all samples,
        revealing important patterns:
        
        **Healthy gradients (ReLU):**
        - Symmetric distribution around zero
        - Moderate spread (-0.1 to +0.1)
        - Stable across epochs
        
        **Vanishing gradients (Sigmoid):**
        - First epoch: Moderate spread
        - Last epoch: Distribution collapsed near zero (all between -0.0001 and +0.0001)
        - Clear shift toward zero over training
        
        **Why signed values matter:**
        - Shows balance of positive/negative updates
        - Reveals asymmetries or biases
        - Can detect dying neurons (all gradients → 0 or one sign dominates)
    """
    # Get layer names
    layer_names = [name for name, _ in model.named_parameters() if 'weight' in name]
    n_layers = len(layer_names)
    n_activations = len(activations_tested)
    
    # Check if per-sample gradients are available
    first_activation = activations_tested[0]
    first_history = gradient_norms_history[first_activation]
    
    if 'per_sample_gradient_distributions' not in first_history:
        logger.warning("Per-sample gradient distributions not found in history!")
        logger.warning("Please train with track_per_sample_gradients=True")
        return
    
    per_sample_grads = first_history['per_sample_gradient_distributions']
    if not per_sample_grads:
        logger.warning("No per-sample gradient data collected!")
        return
    
    # Get epoch indices (should be first, middle, last)
    epoch_indices = sorted(per_sample_grads.keys())
    epoch_names = ['first', 'middle', 'last'][:len(epoch_indices)]
    epoch_titles = ['First Epoch', 'Middle Epoch', 'Last Epoch'][:len(epoch_indices)]
    epoch_colors = ['blue', 'orange', 'red'][:len(epoch_indices)]
    
    # Create a separate figure for each epoch
    for epoch_idx, epoch_name, epoch_title, epoch_color in zip(epoch_indices, epoch_names, epoch_titles, epoch_colors):
        # Create figure with subplots
        fig, axes = plt.subplots(n_activations, n_layers, 
                                figsize=(4 * n_layers, 3.5 * n_activations))
        fig.suptitle(f'Gradient Magnitude Distribution - {epoch_title} (Epoch {epoch_idx + 1})',
                    fontsize=16, fontweight='bold')
        
        # Ensure axes is 2D even if only one activation
        if n_activations == 1:
            axes = axes.reshape(1, -1)
        
        # Plot for each activation and layer
        for act_idx, activation in enumerate(activations_tested):
            history = gradient_norms_history[activation]
            per_sample_grads_at_epoch = history['per_sample_gradient_distributions'][epoch_idx]
            
            for layer_idx, layer_name in enumerate(layer_names):
                ax = axes[act_idx, layer_idx]
                
                # Get per-sample gradient arrays for this layer
                if layer_name in per_sample_grads_at_epoch:
                    gradient_arrays = per_sample_grads_at_epoch[layer_name]
                    
                    # Concatenate all gradient values from all samples
                    all_gradients = np.concatenate(gradient_arrays)
                    
                    if len(all_gradients) > 0:
                        # Plot histogram of signed gradient values
                        ax.hist(all_gradients, bins=50, color=epoch_color, 
                               alpha=0.7, edgecolor='black', linewidth=0.5)
                        
                        # Add statistics
                        mean_grad = np.mean(all_gradients)
                        std_grad = np.std(all_gradients)
                        min_grad = np.min(all_gradients)
                        max_grad = np.max(all_gradients)
                        
                        # Add text with statistics
                        stats_text = (
                            f'Mean: {mean_grad:.2e}\n'
                            f'Std: {std_grad:.2e}\n'
                            f'Range: [{min_grad:.2e}, {max_grad:.2e}]'
                        )
                        ax.text(0.98, 0.97, stats_text,
                               transform=ax.transAxes, fontsize=8,
                               verticalalignment='top', horizontalalignment='right',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        
                        # Add health indicator based on mean absolute value
                        mean_abs_grad = np.mean(np.abs(all_gradients))
                        if mean_abs_grad < 1e-4:
                            status = '⚠️ VANISHED'
                            status_color = 'red'
                        elif mean_abs_grad < 1e-2:
                            status = '⚠ Very Small'
                            status_color = 'orange'
                        elif mean_abs_grad < 0.1:
                            status = 'Small'
                            status_color = 'yellow'
                        else:
                            status = '✓ Healthy'
                            status_color = 'green'
                        
                        ax.text(0.02, 0.97, status,
                               transform=ax.transAxes, fontsize=9,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.5))
                    else:
                        ax.text(0.5, 0.5, 'No gradient data',
                               ha='center', va='center', transform=ax.transAxes,
                               fontsize=10, color='red')
                else:
                    ax.text(0.5, 0.5, 'Layer not found',
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=10, color='red')
                
                # Formatting
                ax.set_xlabel('Gradient Value (signed)', fontsize=10)
                ax.set_ylabel('Count', fontsize=10)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add vertical line at zero
                ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
                
                # Title only on top row
                if act_idx == 0:
                    layer_num = layer_name.split('.')[1] if '.' in layer_name else layer_idx
                    ax.set_title(f'Layer {int(layer_num) + 1}', fontsize=12, fontweight='bold')
                
                # Y-axis label only on leftmost column
                if layer_idx == 0:
                    ax.text(-0.15, 0.5, f'{activation.upper()}', 
                           transform=ax.transAxes,
                           fontsize=12, fontweight='bold', 
                           rotation=90, ha='right', va='center')
        
        # Add overall explanation
        fig.text(0.5, 0.02, 
                f'Histogram shows distribution of SIGNED gradient values across all training samples and parameters\n'
                f'Dashed line = zero | Healthy: Symmetric around zero | Vanishing: Concentrated near zero',
                ha='center', fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.04, 1, 0.97])
        
        # Save with descriptive filename
        filename = f'{save_name}_epoch_{epoch_name}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        logger.info(f"Gradient distribution histogram ({epoch_title}) saved as '{filename}'")
        plt.show()
    
    logger.info(f"Created {len(epoch_indices)} gradient distribution histogram plots")


def plot_regularization_comparison(histories, model_names, save_name='regularization_comparison.png'):
    """
    Compare training with and without regularization using train/validation curves.
    
    Shows:
    1. Train vs Validation Loss (key plot for overfitting detection!)
    2. Overfitting gap (validation - training loss)
    3. Regularization penalty over time
    4. Training stability
    
    Demonstrates how regularization:
    - Prevents overfitting (keeps validation loss close to training loss)
    - Stabilizes training (reduces spikes)
    - Creates smoother loss curves
    
    KEY INSIGHT: The gap between train and validation loss shows overfitting!
    - Small gap → Good generalization
    - Large gap → Overfitting (memorizing training data)
    """
    n_models = len(histories)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Regularization Effects: Train vs. Validation Loss (Overfitting Detection)',
                fontsize=16, fontweight='bold')
    
    # Plot 1: Train vs Validation Loss (MOST IMPORTANT!)
    ax1 = axes[0, 0]
    for name, history in zip(model_names, histories):
        epochs = np.arange(len(history['train_loss']))
        
        # Plot training loss (solid line)
        ax1.plot(epochs, history['train_loss'], linewidth=2.5, alpha=0.9, 
                label=f'{name} (Train)')
        
        # Plot validation loss (dashed line, same color)
        ax1.plot(epochs, history['valid_loss'], '--', linewidth=2.5, alpha=0.9,
                label=f'{name} (Valid)')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Train vs. Validation Loss: Detecting Overfitting', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Add annotation explaining what to look for
    ax1.text(0.02, 0.02, 
            '📊 How to read this plot:\n'
            '• Solid = Training loss\n'
            '• Dashed = Validation loss\n'
            '• Gap widens → Overfitting!\n'
            '• Gap stays small → Good generalization',
            transform=ax1.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            verticalalignment='bottom')
    
    # Plot 2: Overfitting Gap (Validation - Training)
    ax2 = axes[0, 1]
    for name, history in zip(model_names, histories):
        epochs = np.arange(len(history['train_loss']))
        gap = np.array(history['valid_loss']) - np.array(history['train_loss'])
        ax2.plot(epochs, gap, linewidth=2, alpha=0.8, label=name)
    
    ax2.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Overfitting Gap (Valid - Train)', fontsize=12)
    ax2.set_title('Overfitting Gap: Higher = More Overfitting', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add zones
    ax2.axhspan(-0.1, 0.1, alpha=0.1, color='green', label='Good')
    ax2.text(0.02, 0.98, '✓ Green zone: Good generalization\n⚠ Above green: Overfitting!',
            transform=ax2.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='top')
    
    # Plot 3: Regularization penalty
    ax3 = axes[1, 0]
    for name, history in zip(model_names, histories):
        if 'reg_penalty' in history and max(history['reg_penalty']) > 0:
            ax3.plot(history['reg_penalty'], label=name, linewidth=2, alpha=0.8)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Regularization Penalty', fontsize=12)
    ax3.set_title('Regularization Penalty Over Time', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final performance summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    ax4.set_title('Final Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    summary_text = "Final Results (Epoch {}):\n".format(len(history['train_loss']))
    summary_text += "=" * 50 + "\n\n"
    
    for name, history in zip(model_names, histories):
        final_train = history['train_loss'][-1]
        final_valid = history['valid_loss'][-1]
        gap = final_valid - final_train
        
        summary_text += f"{name}:\n"
        summary_text += f"  Train Loss: {final_train:.4f}\n"
        summary_text += f"  Valid Loss: {final_valid:.4f}\n"
        summary_text += f"  Gap: {gap:+.4f}"
        
        if gap < 0.1:
            summary_text += " ✓ Excellent!\n"
        elif gap < 0.5:
            summary_text += " ⚠ Slight overfitting\n"
        else:
            summary_text += " ⚠️ Significant overfitting!\n"
        summary_text += "\n"
    
    summary_text += "\nKEY INSIGHT:\n"
    summary_text += "Regularization keeps validation\n"
    summary_text += "loss close to training loss,\n"
    summary_text += "preventing overfitting!"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    logger.info(f"Regularization comparison (with overfitting analysis) saved as '{save_name}'")
    plt.show()


def plot_results(x_t, y_t,
                 model,
                 training_history,
                 coeffs_true,
                 data_poly_order,
                 model_name="MLP",
                 normalizer=None,
                 show_validation=True):
    """
    Visualize the training results for a single model.
    
    Creates a figure with three subplots:
        1. Data points and fitted curve
        2. Loss curve during training (with validation if available)
        3. Architecture diagram (text-based for MLP)
    
    Args:
        x_t (torch.Tensor)      : Input data (potentially normalized)
        y_t (torch.Tensor)      : Observed output data
        model                   : Trained model (MLP or Perceptron)
        training_history (dict) : Contains loss history
                                 If 'train_loss' and 'valid_loss' present, both shown
                                 If only 'loss' present, training loss only
        coeffs_true (list)      : True polynomial coefficients
        data_poly_order (int)   : Order of true polynomial
        model_name (str)        : Name for plot title
        normalizer              : Optional feature normalizer
        show_validation (bool)  : If True (default), show validation loss if available
                                 If False, only show training loss
    
    Notes:
        - By default, validation loss is shown if present in training_history
        - This helps detect overfitting: gap between train and validation curves
        - Set show_validation=False to only show training loss (legacy behavior)
    """
    # Use GridSpec for better control over subplot sizes
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(20, 6))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.8], wspace=0.3)
    
    # Create subplots with custom sizes
    ax1 = fig.add_subplot(gs[0, 0])  # Data vs. predictions
    ax2 = fig.add_subplot(gs[0, 1])  # Loss curve
    ax3 = fig.add_subplot(gs[0, 2])  # Architecture
    
    # Denormalize for display if needed
    if normalizer is not None:
        x_display = normalizer.inverse_transform(x_t).detach().numpy()
        x_plot_tensor = x_t
    else:
        x_display = x_t.detach().numpy()
        x_plot_tensor = x_t
    
    y_np = y_t.detach().numpy()
    
    # Get model predictions
    with torch.no_grad():
        y_pred = model(x_plot_tensor).numpy()
    
    # Generate dense points for smooth curves
    if normalizer is not None:
        x_dense_original = torch.linspace(x_display.min(), x_display.max(), 500)
        x_dense_normalized = normalizer.transform(x_dense_original)
        
        # True function
        y_true_dense = torch.zeros_like(x_dense_original)
        for i, coeff in enumerate(coeffs_true):
            y_true_dense += coeff * (x_dense_original ** i)
        
        # Model prediction
        with torch.no_grad():
            y_pred_dense = model(x_dense_normalized).numpy()
        
        x_plot = x_dense_original.numpy()
        y_true_plot = y_true_dense.numpy()
        y_pred_plot = y_pred_dense
    else:
        x_plot = x_display
        y_true_plot = torch.zeros_like(x_t)
        for i, coeff in enumerate(coeffs_true):
            y_true_plot += coeff * (x_t ** i)
        y_true_plot = y_true_plot.numpy()
        y_pred_plot = y_pred
    
    # Plot 1: Fitted function
    ax1.scatter(x_display, y_np, alpha=0.5, label='Training data', s=30, c='blue')
    ax1.plot(x_plot, y_pred_plot, 'r-', linewidth=2.5, 
             label=f'{model_name} fit')
    ax1.plot(x_plot, y_true_plot, 'g--', linewidth=2, 
             label=f'True function (order={data_poly_order})')
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title(f'{model_name}: Data vs. Predictions', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss curve (with validation if available and requested)
    has_validation = ('train_loss' in training_history and 
                     'valid_loss' in training_history)
    
    if show_validation and has_validation:
        # Show both training and validation losses
        train_loss = training_history['train_loss']
        valid_loss = training_history['valid_loss']
        epochs = np.arange(len(train_loss))
        
        ax2.plot(epochs, train_loss, 'b-', linewidth=2.5, 
                label='Training Loss', alpha=0.9)
        ax2.plot(epochs, valid_loss, 'r--', linewidth=2.5, 
                label='Validation Loss', alpha=0.9)
        ax2.set_title('Train vs. Validation Loss', fontsize=14, fontweight='bold')
        
        # Add annotation about overfitting
        final_gap = valid_loss[-1] - train_loss[-1]
        if final_gap > 0.5:
            status_text = f'Gap: {final_gap:+.3f}\n⚠️ Overfitting'
            bbox_color = 'red'
        elif final_gap > 0.1:
            status_text = f'Gap: {final_gap:+.3f}\n⚠ Slight overfitting'
            bbox_color = 'orange'
        else:
            status_text = f'Gap: {final_gap:+.3f}\n✓ Good generalization'
            bbox_color = 'green'
        
        ax2.text(0.98, 0.98, status_text,
                transform=ax2.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.3),
                verticalalignment='top', horizontalalignment='right')
    
    else:
        # Show only training loss (legacy format or show_validation=False)
        if 'train_loss' in training_history:
            loss_history = training_history['train_loss']
        else:
            loss_history = training_history.get('loss', [])
        
        epochs = np.arange(len(loss_history))
        ax2.plot(epochs, loss_history, 'b-', linewidth=2, label='Training Loss')
        ax2.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss (MSE)', fontsize=12)
    ax2.set_yscale('log')  # Log scale to see convergence better
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Plot 3: Architecture summary (text-based)
    ax3.axis('off')
    ax3.set_title('Model Architecture', fontsize=14, fontweight='bold', pad=20)
    
    # Get architecture info
    if isinstance(model, MultiLayerPerceptron):
        arch_text = f"Architecture: {' → '.join(map(str, model.layer_sizes))}\n\n"
        arch_text += f"Activation: {model.activation_name}\n\n"
        
        # Add dropout and regularization info
        if hasattr(model, 'dropout_rate') and model.dropout_rate > 0:
            arch_text += f"Dropout: {model.dropout_rate}\n\n"
        
        arch_text += f"Total Parameters: {sum(p.numel() for p in model.parameters())}\n\n"
        arch_text += f"Hidden Layers: {model.num_layers - 1}\n\n"
        
        arch_text += "Layer Details:\n"
        arch_text += "-" * 30 + "\n"
        for i, layer in enumerate(model.layers):
            layer_type = "Hidden" if i < model.num_layers - 1 else "Output"
            arch_text += f"Layer {i+1} ({layer_type}):\n"
            arch_text += f"  {model.layer_sizes[i]} → {model.layer_sizes[i+1]}\n"
            if i < model.num_layers - 1:
                arch_text += f"  + {model.activation_name}() activation\n"
                if hasattr(model, 'dropout_rate') and model.dropout_rate > 0:
                    arch_text += f"  + dropout(p={model.dropout_rate})\n"
    else:
        arch_text = str(model)
    
    # Position text box to fill the subplot area better
    ax3.text(0.05, 0.95, arch_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_results.png', 
                dpi=150, bbox_inches='tight')
    logger.info(f"Results plot saved as '{model_name.lower().replace(' ', '_')}_results.png'")
    plt.show()

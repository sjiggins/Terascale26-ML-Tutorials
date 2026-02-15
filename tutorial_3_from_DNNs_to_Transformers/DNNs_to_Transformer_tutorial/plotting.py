"""
plotting.py
===========
Visualization functions for time-series polynomial forecasting.

This module provides tools to visualize:
1. Time-series evolution of polynomials
2. Forecasting results (history + predictions)
3. Coefficient evolution over time
4. Model comparison plots

Author: ML Tutorial Series
Target Audience: PhD students and early career postdocs
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import logging

logger = logging.getLogger(__name__)


def plot_time_series_evolution(x, y_sequence, coeffs_over_time=None, 
                               title="Polynomial Time-Series Evolution",
                               save_path="time_series_evolution.png"):
    """
    Visualize how the polynomial shape evolves over time.
    
    This creates an educational plot showing:
    - Top: Heatmap of y(x, t) over time
    - Middle: Sample snapshots at different time points
    - Bottom: Coefficient evolution (if provided)
    
    Args:
        x (torch.Tensor): Spatial samples, shape [L]
        y_sequence (torch.Tensor): Time-series data, shape [T, L]
        coeffs_over_time (torch.Tensor, optional): Coefficients, shape [T, k+1]
        title (str): Plot title
        save_path (str): Where to save the figure
    """
    T, L = y_sequence.shape
    
    # Convert to numpy for plotting
    x_np = x.numpy() if torch.is_tensor(x) else x
    y_np = y_sequence.numpy() if torch.is_tensor(y_sequence) else y_sequence
    
    # Create figure
    if coeffs_over_time is not None:
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 1, height_ratios=[2, 2, 1.5], hspace=0.3)
    else:
        fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(2, 1, height_ratios=[2, 2], hspace=0.3)
    
    # ========================================================================
    # PLOT 1: Heatmap of y(x, t)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0])
    
    im = ax1.imshow(
        y_np.T,  # Transpose so x is vertical, time is horizontal
        aspect='auto',
        cmap='RdBu_r',
        interpolation='bilinear',
        extent=[0, T-1, x_np.min(), x_np.max()]
    )
    
    ax1.set_xlabel('Time Step (t)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('x (spatial position)', fontsize=12, fontweight='bold')
    ax1.set_title('Heatmap: How Polynomial Values Change Over Time', 
                 fontsize=14, fontweight='bold', pad=10)
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('y(x, t)', fontsize=11, fontweight='bold')
    
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # ========================================================================
    # PLOT 2: Snapshots at different time points
    # ========================================================================
    ax2 = fig.add_subplot(gs[1])
    
    # Select 6 evenly spaced time points to show
    n_snapshots = 6
    snapshot_indices = np.linspace(0, T-1, n_snapshots, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))
    
    for idx, t in enumerate(snapshot_indices):
        ax2.plot(
            x_np, 
            y_np[t], 
            color=colors[idx],
            linewidth=2,
            alpha=0.8,
            label=f't={t}'
        )
    
    ax2.set_xlabel('x (spatial position)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('y(x, t)', fontsize=12, fontweight='bold')
    ax2.set_title('Polynomial Snapshots at Different Time Points', 
                 fontsize=14, fontweight='bold', pad=10)
    ax2.legend(loc='best', fontsize=10, ncol=3)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # ========================================================================
    # PLOT 3: Coefficient evolution (if provided)
    # ========================================================================
    if coeffs_over_time is not None:
        ax3 = fig.add_subplot(gs[2])
        
        coeffs_np = coeffs_over_time.numpy() if torch.is_tensor(coeffs_over_time) else coeffs_over_time
        k = coeffs_np.shape[1] - 1  # polynomial order
        
        t_axis = np.arange(T)
        colors_coeffs = plt.cm.tab10(np.arange(k+1))
        
        for i in range(k+1):
            ax3.plot(
                t_axis,
                coeffs_np[:, i],
                color=colors_coeffs[i],
                linewidth=2,
                alpha=0.8,
                label=f'a_{i}(t)  (coeff of x^{i})'
            )
        
        ax3.set_xlabel('Time Step (t)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Coefficient Value', fontsize=12, fontweight='bold')
        ax3.set_title('How Polynomial Coefficients Change Over Time', 
                     fontsize=14, fontweight='bold', pad=10)
        ax3.legend(loc='best', fontsize=9, ncol=min(k+1, 5))
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved time-series evolution plot: {save_path}")
    plt.close()


def plot_time_series_plus_forecast(
    x, 
    y_history, 
    y_future_true, 
    y_future_pred=None,
    model_name="Model",
    sample_idx=0,
    title=None,
    save_path="forecast_comparison.png"
):
    """
    Visualize forecasting results: history + true future + predicted future.
    
    This is the KEY visualization for Tutorial 3, showing:
    - Blue region: Historical data the model saw
    - Green line: True future values
    - Red line: Model's predictions (if provided)
    
    Args:
        x (torch.Tensor): Spatial samples, shape [L]
        y_history (torch.Tensor): Historical data, shape [B, T, L] or [T, L]
        y_future_true (torch.Tensor): True future, shape [B, H, L] or [H, L]
        y_future_pred (torch.Tensor, optional): Predicted future, shape [B, H, L] or [H, L]
        model_name (str): Name of the model for labeling
        sample_idx (int): Which sample from batch to plot (if batched)
        title (str): Plot title
        save_path (str): Where to save the figure
    """
    # Handle batched or unbatched inputs
    if y_history.dim() == 3:  # Batched [B, T, L]
        y_hist = y_history[sample_idx]  # [T, L]
        y_true = y_future_true[sample_idx]  # [H, L]
        if y_future_pred is not None:
            y_pred = y_future_pred[sample_idx]  # [H, L]
        else:
            y_pred = None
    else:  # Unbatched [T, L] and [H, L]
        y_hist = y_history
        y_true = y_future_true
        y_pred = y_future_pred
    
    T = y_hist.shape[0]
    H = y_true.shape[0]
    
    # Convert to numpy
    x_np = x.numpy() if torch.is_tensor(x) else x
    y_hist_np = y_hist.numpy() if torch.is_tensor(y_hist) else y_hist
    y_true_np = y_true.numpy() if torch.is_tensor(y_true) else y_true
    if y_pred is not None:
        y_pred_np = y_pred.numpy() if torch.is_tensor(y_pred) else y_pred
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # ========================================================================
    # PLOT 1: Heatmap showing history and future
    # ========================================================================
    ax1 = axes[0]
    
    # Combine history and true future for heatmap
    y_combined = np.concatenate([y_hist_np, y_true_np], axis=0)  # [T+H, L]
    
    im = ax1.imshow(
        y_combined.T,
        aspect='auto',
        cmap='RdBu_r',
        interpolation='bilinear',
        extent=[0, T+H-1, x_np.min(), x_np.max()]
    )
    
    # Draw vertical line separating history from future
    ax1.axvline(x=T-0.5, color='yellow', linestyle='--', linewidth=3, label='Present (t=T)')
    
    # Shade the history region
    ax1.axvspan(0, T-0.5, alpha=0.1, color='blue', label='History (seen by model)')
    
    # Shade the future region
    ax1.axvspan(T-0.5, T+H-1, alpha=0.1, color='green', label='Future (to predict)')
    
    ax1.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax1.set_ylabel('x (spatial position)', fontsize=12, fontweight='bold')
    ax1.set_title('Time-Series Heatmap: History vs Future', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('y(x, t)', fontsize=11, fontweight='bold')
    
    # ========================================================================
    # PLOT 2: Line plots at specific x positions
    # ========================================================================
    ax2 = axes[1]
    
    # Select 3 x positions to show temporal evolution
    L = len(x_np)
    x_positions = [0, L//2, L-1]  # left, center, right
    colors_pos = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
    
    for idx, pos in enumerate(x_positions):
        x_val = x_np[pos]
        
        # Historical values at this x position
        t_hist = np.arange(0, T)
        y_hist_at_x = y_hist_np[:, pos]
        
        # True future at this x position
        t_future = np.arange(T, T+H)
        y_true_at_x = y_true_np[:, pos]
        
        # Plot history (solid line)
        ax2.plot(
            t_hist,
            y_hist_at_x,
            color=colors_pos[idx],
            linewidth=2,
            alpha=0.8,
            label=f'x={x_val:.2f} (history)'
        )
        
        # Plot true future (dashed line)
        ax2.plot(
            t_future,
            y_true_at_x,
            color=colors_pos[idx],
            linewidth=2,
            linestyle='--',
            alpha=0.8,
            label=f'x={x_val:.2f} (true future)'
        )
        
        # Plot predicted future (if available)
        if y_pred is not None:
            y_pred_at_x = y_pred_np[:, pos]
            ax2.plot(
                t_future,
                y_pred_at_x,
                color=colors_pos[idx],
                linewidth=2,
                linestyle=':',
                alpha=0.8,
                marker='o',
                markersize=4,
                label=f'x={x_val:.2f} ({model_name} prediction)'
            )
    
    # Draw vertical line at present
    ax2.axvline(x=T-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Present')
    
    # Shade regions
    ax2.axvspan(0, T-0.5, alpha=0.05, color='blue')
    ax2.axvspan(T-0.5, T+H-1, alpha=0.05, color='green')
    
    ax2.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('y(x, t)', fontsize=12, fontweight='bold')
    ax2.set_title('Temporal Evolution at Selected x Positions', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    if title is None:
        title = f"Forecasting Results: {model_name}"
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved forecast comparison plot: {save_path}")
    plt.close()


def plot_model_comparison(
    x,
    y_history,
    y_future_true,
    predictions_dict,
    sample_idx=0,
    title="Model Comparison: Forecasting Performance",
    save_path="model_comparison.png"
):
    """
    Compare multiple models' forecasting performance.
    
    Args:
        x (torch.Tensor): Spatial samples, shape [L]
        y_history (torch.Tensor): Historical data, shape [B, T, L] or [T, L]
        y_future_true (torch.Tensor): True future, shape [B, H, L] or [H, L]
        predictions_dict (dict): Dictionary mapping model names to predictions
            e.g., {"MLP": y_pred_mlp, "CNN": y_pred_cnn, "Transformer": y_pred_transformer}
        sample_idx (int): Which sample from batch to plot
        title (str): Plot title
        save_path (str): Where to save the figure
    """
    # Handle batched or unbatched inputs
    if y_history.dim() == 3:
        y_hist = y_history[sample_idx]
        y_true = y_future_true[sample_idx]
        preds = {name: pred[sample_idx] for name, pred in predictions_dict.items()}
    else:
        y_hist = y_history
        y_true = y_future_true
        preds = predictions_dict
    
    T = y_hist.shape[0]
    H = y_true.shape[0]
    L = len(x)
    
    # Convert to numpy
    x_np = x.numpy() if torch.is_tensor(x) else x
    y_hist_np = y_hist.numpy() if torch.is_tensor(y_hist) else y_hist
    y_true_np = y_true.numpy() if torch.is_tensor(y_true) else y_true
    preds_np = {name: (pred.numpy() if torch.is_tensor(pred) else pred) 
                for name, pred in preds.items()}
    
    n_models = len(preds)
    
    # Create figure
    fig, axes = plt.subplots(n_models + 1, 1, figsize=(16, 4 * (n_models + 1)))
    if n_models == 0:
        axes = [axes]
    
    # ========================================================================
    # PLOT 1: Ground truth
    # ========================================================================
    ax = axes[0]
    y_combined = np.concatenate([y_hist_np, y_true_np], axis=0)
    
    im = ax.imshow(
        y_combined.T,
        aspect='auto',
        cmap='RdBu_r',
        interpolation='bilinear',
        extent=[0, T+H-1, x_np.min(), x_np.max()]
    )
    
    ax.axvline(x=T-0.5, color='yellow', linestyle='--', linewidth=3)
    ax.axvspan(0, T-0.5, alpha=0.1, color='blue')
    ax.axvspan(T-0.5, T+H-1, alpha=0.1, color='green')
    
    ax.set_ylabel('x', fontsize=11, fontweight='bold')
    ax.set_title('Ground Truth: History + True Future', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('y(x, t)', fontsize=10)
    
    # ========================================================================
    # PLOTS 2+: Each model's predictions
    # ========================================================================
    for idx, (model_name, y_pred_np) in enumerate(preds_np.items(), start=1):
        ax = axes[idx]
        
        # Combine history with this model's prediction
        y_combined = np.concatenate([y_hist_np, y_pred_np], axis=0)
        
        im = ax.imshow(
            y_combined.T,
            aspect='auto',
            cmap='RdBu_r',
            interpolation='bilinear',
            extent=[0, T+H-1, x_np.min(), x_np.max()]
        )
        
        ax.axvline(x=T-0.5, color='yellow', linestyle='--', linewidth=3)
        ax.axvspan(0, T-0.5, alpha=0.1, color='blue')
        ax.axvspan(T-0.5, T+H-1, alpha=0.1, color='orange')
        
        # Compute MSE for this model
        mse = np.mean((y_pred_np - y_true_np) ** 2)
        
        ax.set_ylabel('x', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name} Prediction (MSE: {mse:.4f})', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('y(x, t)', fontsize=10)
    
    axes[-1].set_xlabel('Time Step', fontsize=12, fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved model comparison plot: {save_path}")
    plt.close()


def plot_coefficient_evolution(coeffs_over_time, coeffs_base,
                               title="Polynomial Coefficient Evolution",
                               save_path="coefficient_evolution.png"):
    """
    Plot how polynomial coefficients change over time.
    
    Args:
        coeffs_over_time (torch.Tensor): Time-varying coefficients, shape [T, k+1]
        coeffs_base (list or np.array): Base coefficient values
        title (str): Plot title
        save_path (str): Where to save the figure
    """
    coeffs_np = coeffs_over_time.numpy() if torch.is_tensor(coeffs_over_time) else coeffs_over_time
    T, k_plus_1 = coeffs_np.shape
    k = k_plus_1 - 1
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    t_axis = np.arange(T)
    colors = plt.cm.tab10(np.arange(k+1))
    
    for i in range(k+1):
        ax.plot(
            t_axis,
            coeffs_np[:, i],
            color=colors[i],
            linewidth=2.5,
            alpha=0.8,
            label=f'a_{i}(t)  [base: {coeffs_base[i]:.3f}]'
        )
        
        # Draw horizontal line at base value
        ax.axhline(
            y=coeffs_base[i],
            color=colors[i],
            linestyle='--',
            linewidth=1,
            alpha=0.4
        )
    
    ax.set_xlabel('Time Step (t)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Coefficient Value', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10, ncol=min(k+1, 4))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved coefficient evolution plot: {save_path}")
    plt.close()


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    from logger import configure_logging
    from data_generator import PolynomialTimeSeriesDataset
    
    configure_logging()
    
    logger.info("\n" + "="*70)
    logger.info("DEMONSTRATION: Plotting Functions")
    logger.info("="*70)
    
    # Create sample data
    coeffs_base = [1.0, 0.5, 0.2]
    dataset = PolynomialTimeSeriesDataset(
        coeffs_base=coeffs_base,
        poly_order=2,
        T=100,
        L=50,
        amplitude_scale=0.4,
        seed=42
    )
    
    x, y_seq, coeffs = dataset.generate_sequence(return_coefficients=True)
    
    # Test plot 1: Time-series evolution
    logger.info("\n[TEST 1] Plotting time-series evolution...")
    plot_time_series_evolution(x, y_seq, coeffs)
    
    # Test plot 2: Forecast visualization
    logger.info("\n[TEST 2] Plotting forecast comparison...")
    T_history = 70
    H_forecast = 30
    x_batch, y_hist, y_fut = dataset.generate_batch(1, T_history, H_forecast)
    plot_time_series_plus_forecast(x_batch, y_hist, y_fut, model_name="Example")
    
    # Test plot 3: Coefficient evolution
    logger.info("\n[TEST 3] Plotting coefficient evolution...")
    plot_coefficient_evolution(coeffs, coeffs_base)
    
    logger.info("\n" + "="*70)
    logger.info("All plotting tests completed successfully!")
    logger.info("="*70)

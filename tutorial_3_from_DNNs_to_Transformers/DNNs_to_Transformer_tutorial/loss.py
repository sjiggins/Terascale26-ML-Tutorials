"""
loss.py
=======
Loss functions for sequence-to-sequence forecasting.

This module provides loss functions for training temporal prediction models.
For regression tasks, we primarily use Mean Squared Error (MSE).

Author: ML Tutorial Series
Target Audience: PhD students and early career postdocs
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def mse_loss(y_pred, y_true, reduction='mean'):
    """
    Mean Squared Error loss for sequence-to-sequence prediction.
    
    Mathematical Form:
    ------------------
    For predictions y_pred and targets y_true of shape [B, H, L]:
    
    MSE = (1 / (B × H × L)) Σᵦ Σₕ Σₗ (y_pred[b,h,l] - y_true[b,h,l])²
    
    Where:
    - B: Batch size
    - H: Forecast horizon (number of future time steps)
    - L: Spatial dimension (number of features per time step)
    
    Pedagogical Notes:
    ------------------
    MSE is the standard loss for regression because:
    1. Differentiable everywhere (smooth gradients)
    2. Penalizes large errors more heavily (quadratic)
    3. Has nice statistical properties (maximum likelihood under Gaussian noise)
    4. Easy to interpret (average squared error)
    
    Alternative losses for regression:
    - MAE (L1): More robust to outliers, but non-smooth at zero
    - Huber: Combines MSE and MAE benefits
    - Quantile: For probabilistic forecasting
    
    Args:
        y_pred (torch.Tensor): Predicted sequence, shape [B, H, L]
        y_true (torch.Tensor): True sequence, shape [B, H, L]
        reduction (str): How to reduce loss
            - 'mean': Average over all elements (default)
            - 'sum': Sum over all elements
            - 'none': No reduction, return per-element losses
    
    Returns:
        torch.Tensor: Loss value (scalar if reduction='mean'/'sum')
    """
    return nn.functional.mse_loss(y_pred, y_true, reduction=reduction)


def mae_loss(y_pred, y_true, reduction='mean'):
    """
    Mean Absolute Error loss (L1 loss).
    
    More robust to outliers than MSE, but less smooth.
    
    Args:
        y_pred (torch.Tensor): Predicted sequence, shape [B, H, L]
        y_true (torch.Tensor): True sequence, shape [B, H, L]
        reduction (str): How to reduce loss
    
    Returns:
        torch.Tensor: Loss value
    """
    return nn.functional.l1_loss(y_pred, y_true, reduction=reduction)


def huber_loss(y_pred, y_true, delta=1.0, reduction='mean'):
    """
    Huber loss: MSE for small errors, MAE for large errors.
    
    Combines the benefits of MSE (smooth) and MAE (robust).
    
    Args:
        y_pred (torch.Tensor): Predicted sequence, shape [B, H, L]
        y_true (torch.Tensor): True sequence, shape [B, H, L]
        delta (float): Threshold for switching between quadratic and linear
        reduction (str): How to reduce loss
    
    Returns:
        torch.Tensor: Loss value
    """
    return nn.functional.huber_loss(y_pred, y_true, delta=delta, reduction=reduction)


def rmse_loss(y_pred, y_true):
    """
    Root Mean Squared Error.
    
    RMSE is in the same units as the target variable,
    making it easier to interpret than MSE.
    
    Args:
        y_pred (torch.Tensor): Predicted sequence, shape [B, H, L]
        y_true (torch.Tensor): True sequence, shape [B, H, L]
    
    Returns:
        torch.Tensor: RMSE value
    """
    mse = mse_loss(y_pred, y_true, reduction='mean')
    return torch.sqrt(mse)


def compute_metrics(y_pred, y_true):
    """
    Compute multiple evaluation metrics.
    
    Args:
        y_pred (torch.Tensor): Predicted sequence, shape [B, H, L]
        y_true (torch.Tensor): True sequence, shape [B, H, L]
    
    Returns:
        dict: Dictionary of metrics
    """
    with torch.no_grad():
        metrics = {
            'mse': mse_loss(y_pred, y_true).item(),
            'rmse': rmse_loss(y_pred, y_true).item(),
            'mae': mae_loss(y_pred, y_true).item(),
        }
        
        # Add per-timestep metrics
        # Average over batch and spatial dimensions, keep time dimension
        mse_per_time = mse_loss(y_pred, y_true, reduction='none').mean(dim=[0, 2])
        metrics['mse_per_timestep'] = mse_per_time.cpu().numpy()
        
    return metrics


# ============================================================================
# PEDAGOGICAL DEMONSTRATION
# ============================================================================

def demonstrate_loss_properties():
    """
    Educational demonstration of different loss functions.
    
    Shows how MSE, MAE, and Huber losses behave differently
    for various error magnitudes.
    """
    logger.info("\n" + "="*70)
    logger.info("DEMONSTRATION: Loss Function Properties")
    logger.info("="*70)
    
    # Create sample predictions with different error magnitudes
    errors = torch.tensor([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
    y_true = torch.zeros_like(errors)
    y_pred = errors
    
    logger.info("\nError magnitudes: " + str(errors.tolist()))
    logger.info("\nLoss values:")
    logger.info("-" * 50)
    
    for error in errors:
        y_t = torch.tensor([[error]])
        y_p = torch.tensor([[0.0]])
        
        mse = mse_loss(y_t, y_p).item()
        mae = mae_loss(y_t, y_p).item()
        huber = huber_loss(y_t, y_p, delta=1.0).item()
        
        logger.info(f"Error = {error:5.1f}:  MSE = {mse:6.2f}, "
                   f"MAE = {mae:6.2f}, Huber = {huber:6.2f}")
    
    logger.info("\n" + "="*70)
    logger.info("KEY INSIGHTS:")
    logger.info("  MSE:   Penalizes large errors heavily (quadratic growth)")
    logger.info("  MAE:   Treats all errors equally (linear growth)")
    logger.info("  Huber: Hybrid - quadratic for small, linear for large errors")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    # Demo when run directly
    from logger import configure_logging
    configure_logging()
    
    # Run demonstration
    demonstrate_loss_properties()
    
    # Test loss computation
    logger.info("\nTesting loss computation...")
    y_pred = torch.randn(4, 10, 20)  # [B=4, H=10, L=20]
    y_true = torch.randn(4, 10, 20)
    
    loss = mse_loss(y_pred, y_true)
    logger.info(f"Sample MSE loss: {loss.item():.6f}")
    
    metrics = compute_metrics(y_pred, y_true)
    logger.info(f"Computed metrics: {metrics}")
    logger.info("\n✓ Loss functions working correctly!")

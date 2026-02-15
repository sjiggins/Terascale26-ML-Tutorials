"""
CNN_AR.py
=========
Autoregressive 1D CNN for one-step-ahead forecasting.

This module implements an autoregressive CNN that predicts ONE STEP AT A TIME,
using 1D convolutions to detect local temporal patterns.

KEY DIFFERENCE FROM DIRECT CNN:
-------------------------------
Direct CNN:     [B, T, L] → [B, H, L]  (temporal projection T→H)
Autoregressive: [B, T, L] → [B, 1, L]  (temporal projection T→1)

The convolutional layers are UNCHANGED - they still learn temporal patterns.
Only the final projection changes!

PEDAGOGICAL INSIGHT:
--------------------
Conv filters learn to detect patterns that are useful for prediction.
These patterns are the SAME whether we predict:
- All 50 steps at once (direct)
- One step at a time (autoregressive)

This means conv filters can be TRANSFERRED between models!

Author: ML Tutorial Series
Target Audience: PhD students and early career postdocs
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class CNN_AR(nn.Module):
    """
    Autoregressive 1D CNN for sequential forecasting.
    
    Architecture:
    -------------
    1. Conv1d layers: Extract hierarchical temporal features (UNCHANGED)
    2. Temporal projection: T → 1 (CHANGED from T → H)
    3. Output projection: channels → spatial_dim (UNCHANGED)
    4. Forecast method: Iterative sliding window
    
    Key Insight:
    ------------
    The convolutional filters learn patterns like:
    - Trends (increasing/decreasing)
    - Cycles (periodic patterns)
    - Transitions (regime changes)
    
    These patterns are INDEPENDENT of forecast horizon!
    So conv weights can transfer between direct and AR versions.
    """
    
    def __init__(
        self,
        history_length,
        spatial_dim,
        channels=[64, 128, 64],
        kernel_sizes=[5, 5, 5],
        activation='relu',
        dropout=0.1,
        use_batch_norm=True
    ):
        """
        Initialize autoregressive CNN.
        
        Args:
            history_length (int): T - historical time steps
            spatial_dim (int): L - features per time step
            channels (list): Conv channel dimensions
            kernel_sizes (list): Kernel sizes for each conv layer
            activation (str): Activation function
            dropout (float): Dropout probability
            use_batch_norm (bool): Use batch normalization
        """
        super().__init__()
        
        self.history_length = history_length
        self.spatial_dim = spatial_dim
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.dropout_prob = dropout
        self.use_batch_norm = use_batch_norm
        
        assert len(channels) == len(kernel_sizes)
        
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if use_batch_norm else None
        self.dropout_layers = nn.ModuleList()
        
        in_channels = [spatial_dim] + channels[:-1]
        
        for i, (in_ch, out_ch, k_size) in enumerate(zip(in_channels, channels, kernel_sizes)):
            padding = (k_size - 1) // 2
            conv = nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=k_size,
                padding=padding,
                bias=not use_batch_norm
            )
            self.conv_layers.append(conv)
            
            if use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(out_ch))
            
            if dropout > 0:
                self.dropout_layers.append(nn.Dropout(dropout))
        
        # Temporal projection: T → 1 (CHANGED from T → H)
        self.temporal_projection = nn.Linear(history_length, 1)
        
        # Output projection: channels → spatial_dim
        self.output_projection = nn.Linear(channels[-1], spatial_dim)
        
        self._log_architecture()
    
    def _log_architecture(self):
        """Log architecture details."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        receptive_field = 1
        for k in self.kernel_sizes:
            receptive_field += (k - 1)
        
        logger.info("\n" + "="*70)
        logger.info("AUTOREGRESSIVE 1D CNN Architecture")
        logger.info("="*70)
        logger.info(f"Input shape:   [B, T={self.history_length}, L={self.spatial_dim}]")
        logger.info(f"  → Transpose: [B, L={self.spatial_dim}, T={self.history_length}]")
        
        logger.info(f"\nConvolutional Layers:")
        in_channels = [self.spatial_dim] + self.channels[:-1]
        for i, (in_ch, out_ch, k_size) in enumerate(zip(in_channels, self.channels, self.kernel_sizes)):
            logger.info(f"  Layer {i+1}: Conv1d({in_ch:3d} → {out_ch:3d}, kernel={k_size})")
            if self.use_batch_norm:
                logger.info(f"           + BatchNorm1d({out_ch})")
            logger.info(f"           + {self.activation.__class__.__name__}")
            if self.dropout_prob > 0:
                logger.info(f"           + Dropout({self.dropout_prob})")
        
        logger.info(f"\nTemporal Projection:")
        logger.info(f"  Linear(T={self.history_length} → 1)  ← AUTOREGRESSIVE")
        logger.info(f"  Predicts only NEXT time step")
        
        logger.info(f"\nOutput Projection:")
        logger.info(f"  Linear({self.channels[-1]} → {self.spatial_dim})")
        
        logger.info(f"\n  → Output: [B, 1, L={self.spatial_dim}]  ← SINGLE STEP")
        
        logger.info(f"\nReceptive field: {receptive_field} time steps")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info("\n⚠️  AUTOREGRESSIVE MODE:")
        logger.info("   Use forecast() method for multi-step prediction")
        logger.info("="*70 + "\n")
    
    def forward(self, x):
        """
        Forward pass: predict next time step.
        
        Args:
            x (torch.Tensor): Input history, shape [B, T, L]
        
        Returns:
            torch.Tensor: Next step prediction, shape [B, 1, L]
        """
        # Transpose for Conv1d
        x = x.transpose(1, 2)  # [B, T, L] → [B, L, T]
        
        # Convolutional layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            
            if self.use_batch_norm:
                x = self.bn_layers[i](x)
            
            x = self.activation(x)
            
            if self.dropout_prob > 0:
                x = self.dropout_layers[i](x)
        
        # After convs: [B, channels[-1], T]
        
        # Temporal projection: T → 1
        x = self.temporal_projection(x)  # [B, channels[-1], 1]
        
        # Transpose
        x = x.transpose(1, 2)  # [B, 1, channels[-1]]
        
        # Output projection
        x = self.output_projection(x)  # [B, 1, L]
        
        return x
    
    def forecast(self, x_history, n_steps, return_all_steps=True):
        """
        Autoregressive multi-step forecasting.
        
        Args:
            x_history (torch.Tensor): Initial history, shape [B, T, L]
            n_steps (int): Number of steps to forecast
            return_all_steps (bool): Whether to return all predictions
        
        Returns:
            torch.Tensor: Forecasted sequence, shape [B, n_steps, L]
        """
        predictions = []
        current_history = x_history.clone()
        
        for step in range(n_steps):
            # Predict next step
            y_next = self.forward(current_history)  # [B, 1, L]
            predictions.append(y_next)
            
            # Slide window
            current_history = torch.cat([
                current_history[:, 1:, :],
                y_next
            ], dim=1)
        
        return torch.cat(predictions, dim=1)  # [B, n_steps, L]
    
    def count_parameters(self):
        """Count parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    from logger import configure_logging
    configure_logging()
    
    logger.info("Creating autoregressive CNN...")
    model = CNN_AR(
        history_length=50,
        spatial_dim=100,
        channels=[64, 128, 64],
        kernel_sizes=[5, 5, 5]
    )
    
    # Test single-step
    logger.info("\n[TEST 1] Single-step prediction:")
    x_test = torch.randn(8, 50, 100)
    y_next = model(x_test)
    logger.info(f"Input:  {x_test.shape}")
    logger.info(f"Output: {y_next.shape}")
    logger.info(f"✓ Single-step successful!")
    
    # Test multi-step
    logger.info("\n[TEST 2] Multi-step forecasting:")
    y_forecast = model.forecast(x_test, n_steps=10)
    logger.info(f"Forecast: {y_forecast.shape}")
    logger.info(f"✓ Multi-step successful!")

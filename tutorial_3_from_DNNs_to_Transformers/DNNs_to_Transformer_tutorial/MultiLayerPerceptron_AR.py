"""
MultiLayerPerceptron_AR.py
===========================
Autoregressive Multi-Layer Perceptron for one-step-ahead forecasting.

This module implements an autoregressive MLP that predicts ONE STEP AT A TIME,
feeding predictions back as input for the next prediction (like GPT for text).

KEY DIFFERENCE FROM DIRECT MLP:
-------------------------------
Direct MLP:     [B, T, L] → [B, H, L]  (predicts all H steps at once)
Autoregressive: [B, T, L] → [B, 1, L]  (predicts only next step)
                Then loops H times, feeding predictions back

PEDAGOGICAL CONCEPT:
--------------------
This is how GPT-style models work:
1. Given context → predict next token
2. Append prediction to context
3. Predict next token
4. Repeat...

In our case:
- "Tokens" = time steps
- "Context" = sliding window of past observations

Author: ML Tutorial Series
Target Audience: PhD students and early career postdocs
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class MultiLayerPerceptronAR(nn.Module):
    """
    Autoregressive MLP for sequential one-step-ahead forecasting.
    
    Architecture Change:
    --------------------
    Output layer predicts ONLY the next time step (not all H steps):
    - Direct MLP: Linear(hidden, T*L) → reshape to [B, H, L]
    - AR MLP: Linear(hidden, L) → [B, 1, L]
    
    Multi-Step Forecasting:
    -----------------------
    Achieved through the forecast() method which iteratively:
    1. Predicts next step
    2. Slides window (drops oldest, adds newest)
    3. Repeats H times
    
    Advantages:
    -----------
    - Can adapt predictions based on previous predictions
    - Natural for sequential generation tasks
    - More like how humans think: "What happens next?"
    
    Disadvantages:
    --------------
    - SLOW: Requires H forward passes (vs. 1 for direct)
    - Error accumulation: mistakes compound
    - Training more complex (teacher forcing vs. autoregressive)
    """
    
    def __init__(
        self,
        history_length,
        spatial_dim,
        hidden_dims=[512, 256, 128],
        activation='relu',
        dropout=0.1,
        use_batch_norm=False
    ):
        """
        Initialize autoregressive MLP.
        
        Args:
            history_length (int): T - number of historical time steps to observe
            spatial_dim (int): L - number of features per time step
            hidden_dims (list): Hidden layer dimensions
            activation (str): Activation function
            dropout (float): Dropout probability
            use_batch_norm (bool): Whether to use batch normalization
            
        Note: No forecast_horizon parameter! Always predicts 1 step ahead.
        """
        super().__init__()
        
        self.history_length = history_length
        self.spatial_dim = spatial_dim
        self.hidden_dims = hidden_dims
        self.dropout_prob = dropout
        self.use_batch_norm = use_batch_norm
        
        # Input dimension: flatten history
        self.input_dim = history_length * spatial_dim
        
        # Output dimension: ONLY next time step
        self.output_dim = spatial_dim  # Just L, not H*L
        
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network
        layers = []
        dims = [self.input_dim] + hidden_dims + [self.output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            if use_batch_norm and i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            
            if i < len(dims) - 2:
                layers.append(self.activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
        
        self._log_architecture()
    
    def _log_architecture(self):
        """Log architecture details."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info("\n" + "="*70)
        logger.info("AUTOREGRESSIVE MLP Architecture")
        logger.info("="*70)
        logger.info(f"Input shape:  [B, T={self.history_length}, L={self.spatial_dim}]")
        logger.info(f"  → Flatten:  [B, {self.input_dim}]")
        logger.info(f"\nHidden layers: {self.hidden_dims}")
        logger.info(f"Activation: {self.activation.__class__.__name__}")
        logger.info(f"Dropout: {self.dropout_prob}")
        logger.info(f"Batch Norm: {self.use_batch_norm}")
        logger.info(f"\n  → Output:   [B, {self.output_dim}]")
        logger.info(f"  → Reshape:  [B, 1, L={self.spatial_dim}]  ← ONLY NEXT STEP")
        logger.info(f"\nTotal parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info("\n⚠️  AUTOREGRESSIVE MODE:")
        logger.info("   Multi-step forecasting requires calling forecast() method")
        logger.info("   which iteratively predicts and slides the window.")
        logger.info("="*70 + "\n")
    
    def forward(self, x):
        """
        Forward pass: predict ONLY the next time step.
        
        Args:
            x (torch.Tensor): Input history, shape [B, T, L]
        
        Returns:
            torch.Tensor: Next step prediction, shape [B, 1, L]
        """
        batch_size = x.shape[0]
        
        # Flatten
        x_flat = x.view(batch_size, -1)  # [B, T*L]
        
        # MLP forward
        y_flat = self.network(x_flat)  # [B, L]
        
        # Reshape to sequence format
        y_next = y_flat.view(batch_size, 1, self.spatial_dim)  # [B, 1, L]
        
        return y_next
    
    def forecast(self, x_history, n_steps, return_all_steps=True):
        """
        Autoregressive multi-step forecasting.
        
        This is the KEY method for autoregressive models!
        
        Args:
            x_history (torch.Tensor): Initial history, shape [B, T, L]
            n_steps (int): Number of future steps to predict (H)
            return_all_steps (bool): If True, return all predictions.
                                     If False, return only final predictions.
        
        Returns:
            torch.Tensor: Forecasted sequence, shape [B, H, L]
        
        Algorithm:
        ----------
        for step in range(n_steps):
            1. y_next = forward(current_history)  # Predict next
            2. current_history = slide_window(current_history, y_next)  # Update
            3. store y_next
        return all predictions
        """
        predictions = []
        current_history = x_history.clone()  # [B, T, L]
        
        for step in range(n_steps):
            # Predict next step
            y_next = self.forward(current_history)  # [B, 1, L]
            predictions.append(y_next)
            
            # Slide window: drop oldest time step, append newest prediction
            current_history = torch.cat([
                current_history[:, 1:, :],  # Remove t=0, keep t=1...T-1
                y_next                       # Add predicted t=T
            ], dim=1)  # New shape: [B, T, L]
        
        # Concatenate all predictions
        return torch.cat(predictions, dim=1)  # [B, H, L]
    
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
    
    logger.info("Creating autoregressive MLP...")
    model = MultiLayerPerceptronAR(
        history_length=50,
        spatial_dim=100,
        hidden_dims=[512, 256, 128]
    )
    
    # Test single-step prediction
    logger.info("\n[TEST 1] Single-step prediction:")
    x_test = torch.randn(8, 50, 100)
    y_next = model(x_test)
    logger.info(f"Input:  {x_test.shape}")
    logger.info(f"Output: {y_next.shape}")
    logger.info(f"✓ Single-step forward pass successful!")
    
    # Test multi-step forecasting
    logger.info("\n[TEST 2] Multi-step autoregressive forecasting:")
    y_forecast = model.forecast(x_test, n_steps=10)
    logger.info(f"Input:    {x_test.shape}")
    logger.info(f"Forecast: {y_forecast.shape}")
    logger.info(f"✓ Multi-step forecast successful!")
    
    # Compare parameter counts
    total, trainable = model.count_parameters()
    logger.info(f"\nModel parameters: {total:,}")
    logger.info(f"Compare to direct MLP with forecast_horizon=10:")
    logger.info(f"  Direct MLP output: Linear(128, 10*100) = 128,000 parameters in output layer")
    logger.info(f"  AR MLP output:     Linear(128, 100)     = 12,800 parameters in output layer")
    logger.info(f"  Savings: ~115,200 parameters! But requires 10 forward passes.")

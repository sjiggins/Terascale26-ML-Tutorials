"""
Transformer_AR.py
=================
Autoregressive Transformer for one-step-ahead forecasting.

This module implements a GPT-style autoregressive Transformer!

KEY INSIGHT:
------------
This is EXACTLY how GPT works:
1. Given context (past tokens/time steps)
2. Predict next token/time step
3. Append prediction to context
4. Repeat

The only differences from GPT:
- We use continuous values (not discrete tokens)
- We use sinusoidal positional encoding (not learned)
- We're doing regression (not classification)

But the ARCHITECTURE and GENERATION PROCESS are identical!

PEDAGOGICAL GOLD:
-----------------
After completing this tutorial, students can say:
"I built GPT!" (Well, a simplified version for time-series)

Author: ML Tutorial Series
Target Audience: PhD students and early career postdocs
"""

import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (same as before).
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerAR(nn.Module):
    """
    Autoregressive Transformer for sequential forecasting.
    
    This is a GPT-style decoder-only Transformer!
    
    Architecture Flow:
    ------------------
    Input: [B, T, L]
      ↓
    Project to d_model: [B, T, d_model]
      ↓
    Add positional encoding
      ↓
    Transformer encoder layers (self-attention)
      ↓
    Take ONLY last position: [B, 1, d_model]
      ↓
    Project to output: [B, 1, L]
    
    Key Difference from Direct:
    ---------------------------
    - Direct: Uses ALL T positions' outputs → project to H steps
    - AR: Uses ONLY last position's output → predicts next step
    
    Why Last Position?
    ------------------
    After self-attention, the last position's representation contains:
    - Information from ALL previous positions (via attention)
    - The "summary" of the full context
    - Everything needed to predict what comes next
    
    This is exactly how GPT works!
    """
    
    def __init__(
        self,
        history_length,
        spatial_dim,
        d_model=128,
        n_heads=4,
        n_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        activation='relu'
    ):
        """
        Initialize autoregressive Transformer.
        
        Args:
            history_length (int): T - context length
            spatial_dim (int): L - input features
            d_model (int): Transformer hidden dimension
            n_heads (int): Number of attention heads
            n_layers (int): Number of transformer layers
            dim_feedforward (int): FFN dimension
            dropout (float): Dropout probability
            activation (str): Activation function
        """
        super().__init__()
        
        self.history_length = history_length
        self.spatial_dim = spatial_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_prob = dropout
        
        # Input projection
        self.input_projection = nn.Linear(spatial_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=history_length * 2,  # Allow for growing context
            dropout=dropout
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # Output projection: ONLY for next step
        self.output_projection = nn.Linear(d_model, spatial_dim)
        
        self._init_weights()
        self._log_architecture()
    
    def _init_weights(self):
        """Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _log_architecture(self):
        """Log architecture details."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info("\n" + "="*70)
        logger.info("AUTOREGRESSIVE TRANSFORMER (GPT-style!)")
        logger.info("="*70)
        logger.info(f"Input shape:  [B, T={self.history_length}, L={self.spatial_dim}]")
        
        logger.info(f"\nInput Projection:")
        logger.info(f"  Linear(L={self.spatial_dim} → d_model={self.d_model})")
        
        logger.info(f"\nPositional Encoding:")
        logger.info(f"  Sinusoidal encoding (max_len={self.history_length*2})")
        logger.info(f"  → [B, T, d_model={self.d_model}]")
        
        logger.info(f"\nTransformer Encoder:")
        logger.info(f"  Number of layers: {self.n_layers}")
        logger.info(f"  d_model: {self.d_model}")
        logger.info(f"  Number of heads: {self.n_heads}")
        logger.info(f"  d_k per head: {self.d_model // self.n_heads}")
        logger.info(f"  FFN dimension: {self.dim_feedforward}")
        logger.info(f"  Dropout: {self.dropout_prob}")
        logger.info(f"  → Self-attention across ALL T positions")
        
        logger.info(f"\n⭐ AUTOREGRESSIVE OUTPUT:")
        logger.info(f"  Take ONLY last position: [:, -1:, :]")
        logger.info(f"  → [B, 1, d_model={self.d_model}]")
        
        logger.info(f"\nOutput Projection:")
        logger.info(f"  Linear(d_model={self.d_model} → L={self.spatial_dim})")
        logger.info(f"  → [B, 1, L={self.spatial_dim}]  ← NEXT STEP ONLY")
        
        logger.info(f"\nTotal parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        logger.info(f"\n🎯 THIS IS GPT FOR TIME-SERIES!")
        logger.info(f"   - GPT predicts next word given context")
        logger.info(f"   - This predicts next time step given history")
        logger.info(f"   - Same architecture, different domain!")
        logger.info("="*70 + "\n")
    
    def forward(self, x):
        """
        Forward pass: predict next time step.
        
        Args:
            x (torch.Tensor): Input history, shape [B, T, L]
        
        Returns:
            torch.Tensor: Next step prediction, shape [B, 1, L]
        """
        # Input projection
        x = self.input_projection(x)  # [B, T, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)  # [B, T, d_model]
        
        # Self-attention across all positions
        x = self.transformer_encoder(x)  # [B, T, d_model]
        
        # ⭐ KEY: Take ONLY the last position's output
        # This position has attended to all previous positions
        x_last = x[:, -1:, :]  # [B, 1, d_model]
        
        # Project to output space
        y_next = self.output_projection(x_last)  # [B, 1, L]
        
        return y_next
    
    def forecast(self, x_history, n_steps, return_all_steps=True):
        """
        Autoregressive multi-step forecasting (GPT-style generation!).
        
        This is EXACTLY how GPT generates text:
        1. Given context → predict next token
        2. Append token to context
        3. Predict next token
        4. Repeat...
        
        Args:
            x_history (torch.Tensor): Initial context, shape [B, T, L]
            n_steps (int): Number of steps to generate
            return_all_steps (bool): Return all predictions
        
        Returns:
            torch.Tensor: Generated sequence, shape [B, n_steps, L]
        """
        predictions = []
        current_history = x_history.clone()
        
        for step in range(n_steps):
            # Predict next step (like GPT predicting next word)
            y_next = self.forward(current_history)  # [B, 1, L]
            predictions.append(y_next)
            
            # Append prediction to context (growing context)
            current_history = torch.cat([
                current_history[:, 1:, :],  # Slide window
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
    
    logger.info("Creating GPT-style autoregressive Transformer...")
    model = TransformerAR(
        history_length=50,
        spatial_dim=100,
        d_model=128,
        n_heads=4,
        n_layers=3
    )
    
    # Test single-step
    logger.info("\n[TEST 1] Single-step prediction (like GPT next-token):")
    x_test = torch.randn(8, 50, 100)
    y_next = model(x_test)
    logger.info(f"Context: {x_test.shape}")
    logger.info(f"Next:    {y_next.shape}")
    logger.info(f"✓ Next-step prediction successful!")
    
    # Test multi-step
    logger.info("\n[TEST 2] Multi-step generation (like GPT text generation):")
    y_forecast = model.forecast(x_test, n_steps=10)
    logger.info(f"Context:   {x_test.shape}")
    logger.info(f"Generated: {y_forecast.shape}")
    logger.info(f"✓ Multi-step generation successful!")
    
    logger.info("\n" + "="*70)
    logger.info("🎉 CONGRATULATIONS!")
    logger.info("You've just built a GPT-style autoregressive Transformer!")
    logger.info("="*70)

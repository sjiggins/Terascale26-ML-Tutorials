"""
CNN_AR_v2.py
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class CNN_AR_v2(nn.Module):
    """
    Autoregressive CNN
    """
    
    def __init__(
        self,
        history_length,
        spatial_dim,
        channels=[64, 128, 64],
        kernel_sizes=[5, 5, 5],
        activation='relu',
        dropout=0.1,
        normalization='none',  # 'none', 'instance', 'layer'
        aggregation='adaptive_avg'
    ):
        super().__init__()
        
        self.history_length = history_length
        self.spatial_dim = spatial_dim
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.dropout_prob = dropout
        self.normalization = normalization
        self.aggregation = aggregation
        
        assert len(channels) == len(kernel_sizes)
        assert aggregation in ['adaptive_avg', 'adaptive_max', 'last_position', 'attention']
        assert normalization in ['none', 'instance', 'layer']
        
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
        self.norm_layers = nn.ModuleList() if normalization != 'none' else None
        self.dropout_layers = nn.ModuleList()
        
        in_channels = [spatial_dim] + channels[:-1]
        
        for i, (in_ch, out_ch, k_size) in enumerate(zip(in_channels, channels, kernel_sizes)):
            padding = (k_size - 1) // 2
            conv = nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=k_size,
                padding=padding,
                bias=True  # Always use bias (no BatchNorm)
            )
            self.conv_layers.append(conv)
            
            # Normalization (if any)
            if normalization == 'instance':
                self.norm_layers.append(nn.InstanceNorm1d(out_ch))
            elif normalization == 'layer':
                self.norm_layers.append(nn.LayerNorm(out_ch))
            
            if dropout > 0:
                self.dropout_layers.append(nn.Dropout(dropout))
        
        # Temporal aggregation
        if aggregation == 'adaptive_avg':
            self.temporal_aggregation = nn.AdaptiveAvgPool1d(1)
        elif aggregation == 'adaptive_max':
            self.temporal_aggregation = nn.AdaptiveMaxPool1d(1)
        elif aggregation == 'last_position':
            self.temporal_aggregation = None
        elif aggregation == 'attention':
            self.attention_weights = nn.Linear(channels[-1], 1)
            self.temporal_aggregation = None
        
        # Output projection
        self.output_projection = nn.Linear(channels[-1], spatial_dim)
        
        self._log_architecture()
    
    def _log_architecture(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        receptive_field = 1
        for k in self.kernel_sizes:
            receptive_field += (k - 1)
        
        logger.info("\n" + "="*70)
        logger.info("FIXED AUTOREGRESSIVE CNN (NO BATCHNORM!)")
        logger.info("="*70)
        logger.info(f"Input shape:   [B, T={self.history_length}, L={self.spatial_dim}]")
        
        logger.info(f"\nConvolutional Layers:")
        in_channels = [self.spatial_dim] + self.channels[:-1]
        for i, (in_ch, out_ch, k_size) in enumerate(zip(in_channels, self.channels, self.kernel_sizes)):
            logger.info(f"  Layer {i+1}: Conv1d({in_ch:3d} → {out_ch:3d}, kernel={k_size})")
            if self.normalization != 'none':
                logger.info(f"           + {self.normalization.capitalize()}Norm")
            logger.info(f"           + {self.activation.__class__.__name__}")
            if self.dropout_prob > 0:
                logger.info(f"           + Dropout({self.dropout_prob})")
        
        logger.info(f"\n FIXED Temporal Aggregation:")
        logger.info(f"  Method: {self.aggregation}")
        
        logger.info(f"\nOutput Projection:")
        logger.info(f"  Linear({self.channels[-1]} → {self.spatial_dim})")
        
        logger.info(f"\nReceptive field: {receptive_field} time steps")
        logger.info(f"Total parameters: {total_params:,}")
        
        logger.info("\n FIX SUMMARY:")
        logger.info(f"   Normalization: {self.normalization} (NOT BatchNorm!)")
        logger.info("   Why: BatchNorm stats mismatch in autoregressive mode")
        logger.info("   Training sees TRUE data, generation sees PREDICTIONS")
        logger.info("   This distribution shift breaks BatchNorm!")
        logger.info("="*70 + "\n")
    
    def forward(self, x):
        # x: [B, T, L]
        x = x.transpose(1, 2)  # [B, L, T]
        
        # Convolutional layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            
            # Normalization
            if self.normalization == 'instance':
                x = self.norm_layers[i](x)
            elif self.normalization == 'layer':
                # LayerNorm expects [B, T, C] format
                x = x.transpose(1, 2)  # [B, T, C]
                x = self.norm_layers[i](x)
                x = x.transpose(1, 2)  # [B, C, T]
            
            x = self.activation(x)
            
            if self.dropout_prob > 0:
                x = self.dropout_layers[i](x)
        
        # After convs: [B, channels[-1], T]
        
        # Temporal aggregation
        if self.aggregation == 'adaptive_avg' or self.aggregation == 'adaptive_max':
            x = self.temporal_aggregation(x)  # [B, channels, 1]
            
        elif self.aggregation == 'last_position':
            x = x[:, :, -1:]  # [B, channels, 1]
            
        elif self.aggregation == 'attention':
            x_transposed = x.transpose(1, 2)  # [B, T, channels]
            attention_scores = self.attention_weights(x_transposed)  # [B, T, 1]
            attention_weights = torch.softmax(attention_scores, dim=1)
            x = torch.sum(x * attention_weights.transpose(1, 2), dim=2, keepdim=True)
        
        # x: [B, channels, 1]
        x = x.transpose(1, 2)  # [B, 1, channels]
        x = self.output_projection(x)  # [B, 1, L]
        
        return x
    
    def forecast(self, x_history, n_steps, return_all_steps=True):
        predictions = []
        current_history = x_history.clone()
        
        for step in range(n_steps):
            y_next = self.forward(current_history)
            predictions.append(y_next)
            
            current_history = torch.cat([
                current_history[:, 1:, :],
                y_next
            ], dim=1)
        
        return torch.cat(predictions, dim=1)
    
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


if __name__ == "__main__":
    from logger import configure_logging
    configure_logging()
    
    logger.info("CNN AR")
    model = CNN_AR_v2_Fixed(
        history_length=150,
        spatial_dim=100,
        channels=[64, 128, 64],
        kernel_sizes=[5, 5, 5],
        normalization='none',
        aggregation='adaptive_avg'
    )
    
    x_test = torch.randn(8, 150, 100)
    y_next = model(x_test)
    logger.info(f" Single step: {x_test.shape} → {y_next.shape}")
    
    y_forecast = model.forecast(x_test, n_steps=50)
    logger.info(f" Multi-step: {x_test.shape} → {y_forecast.shape}")

"""
MultiLayerPerceptron.py
=======================
Neural network models: Single Perceptron and Multi-Layer Perceptron (MLP).

This module defines neural network architectures that can learn to approximate
arbitrary functions, including polynomial relationships.

Key Concepts:
    - Perceptron: Single linear transformation (no hidden layers)
    - MLP: Stack of linear transformations with non-linear activations
    - Universal Approximation: MLPs can approximate any continuous function

Author: ML Tutorial Series
Target Audience: PhD students and early career postdocs
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class SinglePerceptron(nn.Module):
    """
    A single perceptron (single-layer neural network).
    
    Architecture:
        Input (1D) → Linear Layer → Output (1D)
        
        Mathematically: y = w·x + b
        
    This is equivalent to linear regression but uses the nn.Module framework,
    preparing students for more complex architectures.
    
    Attributes:
        input_dim (int): Dimension of input features (default: 1)
        output_dim (int): Dimension of output (default: 1)
        linear (nn.Linear): The single linear transformation layer
    
    Pedagogical Note:
        A single perceptron WITHOUT activation function is exactly equivalent
        to linear regression. This demonstrates that linear regression is a
        special case of neural networks.
    """
    
    def __init__(self, input_dim=1, output_dim=1):
        """
        Initialize the single perceptron.
        
        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of output units (typically 1 for regression)
        """
        super(SinglePerceptron, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Single linear layer: y = Wx + b
        # - W (weight matrix): shape (output_dim, input_dim)
        # - b (bias vector): shape (output_dim,)
        self.linear = nn.Linear(input_dim, output_dim)
        
        logger.info(f"Initialized SinglePerceptron: {input_dim} → {output_dim}")
        logger.info(f"Parameters: W shape = {list(self.linear.weight.shape)}, "
                   f"b shape = {list(self.linear.bias.shape)}")
    
    def forward(self, x):
        """
        Forward pass through the perceptron.
        
        Args:
            x (torch.Tensor): Input tensor of shape (n_samples, input_dim)
                             or (n_samples,) which will be reshaped
        
        Returns:
            torch.Tensor: Output predictions of shape (n_samples, output_dim)
        
        Mathematical Operation:
            y = W·x + b
            
        where:
            - W: weight matrix (learned)
            - b: bias vector (learned)
            - x: input features
            - y: predicted output
        """
        # Ensure input has correct shape
        if x.dim() == 1:
            x = x.unsqueeze(1)  # (n_samples,) → (n_samples, 1)
        
        # Apply linear transformation
        y_pred = self.linear(x)
        
        # For single output, flatten to (n_samples,)
        if self.output_dim == 1:
            y_pred = y_pred.squeeze(1)
        
        return y_pred
    
    def get_parameters(self):
        """
        Get the weight and bias parameters.
        
        Returns:
            dict: {'weight': weight_values, 'bias': bias_values}
        """
        return {
            'weight': self.linear.weight.detach().cpu().numpy(),
            'bias': self.linear.bias.detach().cpu().numpy()
        }
    
    def __repr__(self):
        """String representation of the perceptron."""
        params = self.get_parameters()
        w = params['weight'].flatten()
        b = params['bias'].flatten()
        
        if len(w) == 1:
            return f"SinglePerceptron(y = {w[0]:.4f}·x + {b[0]:.4f})"
        else:
            return f"SinglePerceptron(input_dim={self.input_dim}, output_dim={self.output_dim})"


class MultiLayerPerceptron(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with configurable architecture.
    
    Architecture:
        Input → Hidden Layer 1 → ... → Hidden Layer N → Output
        
    Each layer consists of:
        1. Linear transformation: z = W·x + b
        2. Non-linear activation: a = σ(z)
    
    Key Concept - Universal Approximation Theorem:
        An MLP with:
        - At least one hidden layer
        - Non-linear activation function
        - Sufficient hidden units
        
        can approximate ANY continuous function on a compact subset of ℝⁿ
        to arbitrary precision.
    
    Attributes:
        layer_sizes (list): List of layer dimensions [input, hidden1, ..., output]
        activation (str): Activation function name ('relu', 'tanh', 'sigmoid')
        use_activation_output (bool): Apply activation to output layer
        layers (nn.ModuleList): List of linear layers
        activation_fn (nn.Module): Activation function instance
    
    Example:
        # MLP with architecture: 1 → 64 → 32 → 1
        model = MultiLayerPerceptron([1, 64, 32, 1], activation='relu')
        
        This creates:
        - Input layer: 1D
        - Hidden layer 1: 64 units with ReLU
        - Hidden layer 2: 32 units with ReLU
        - Output layer: 1D (no activation by default for regression)
    """
    
    def __init__(self, layer_sizes, activation='relu', use_activation_output=False, dropout_rate=0.0):
        """
        Initialize the Multi-Layer Perceptron.
        
        Args:
            layer_sizes (list): Dimensions of each layer [input, hidden1, ..., output]
                               Example: [1, 64, 32, 1] creates 1→64→32→1 architecture
            activation (str): Activation function type
                - 'relu': ReLU(x) = max(0, x) - most common, prevents vanishing gradients
                - 'tanh': tanh(x) - outputs in [-1, 1], zero-centered
                - 'sigmoid': σ(x) = 1/(1+e^(-x)) - outputs in [0, 1]
                - 'none': No activation (linear network)
            use_activation_output (bool): Whether to apply activation to output layer
                For regression: False (linear output)
                For classification: True (e.g., sigmoid for binary)
            dropout_rate (float): Dropout probability (0 = no dropout, 0.5 = 50% dropout)
                Dropout is a regularization technique that randomly sets
                a fraction of neurons to zero during training.
                
                Why dropout works:
                - Prevents co-adaptation of neurons (neurons can't rely on each other)
                - Acts like training an ensemble of networks
                - Reduces overfitting
                
                Typical values:
                - 0.0: No dropout (default)
                - 0.2-0.3: Light regularization
                - 0.5: Standard dropout
                - 0.7-0.8: Heavy regularization (rare)
        
        Raises:
            ValueError: If layer_sizes has less than 2 elements or dropout_rate invalid
        """
        super(MultiLayerPerceptron, self).__init__()
        
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements (input and output)")
        
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.use_activation_output = use_activation_output
        self.num_layers = len(layer_sizes) - 1  # Number of linear transformations
        self.dropout_rate = dropout_rate
        
        # Build the network architecture
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            logger.debug(f"Layer {i+1}: {layer_sizes[i]} → {layer_sizes[i+1]}")
        
        # Select activation function
        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation_fn = nn.Sigmoid()
        elif activation == 'none':
            self.activation_fn = nn.Identity()  # No activation (linear)
        else:
            raise ValueError(f"Unknown activation: {activation}. "
                           f"Choose from: 'relu', 'tanh', 'sigmoid', 'none'")
        
        # Dropout layer (only active during training)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
            logger.debug(f"Dropout enabled: p={dropout_rate}")
        else:
            self.dropout = None
        
        # Count total parameters
        total_params = sum(p.numel() for p in self.parameters())
        
        logger.info(f"Initialized MultiLayerPerceptron:")
        logger.info(f"  Architecture: {' → '.join(map(str, layer_sizes))}")
        logger.info(f"  Activation: {activation}")
        logger.info(f"  Dropout rate: {dropout_rate}")
        logger.info(f"  Total parameters: {total_params}")
        logger.info(f"  Hidden layers: {self.num_layers - 1}")
    
    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Computes:
            h₁ = σ(W₁·x + b₁)           # First hidden layer
            h₁ = dropout(h₁)            # Apply dropout (training only)
            h₂ = σ(W₂·h₁ + b₂)          # Second hidden layer
            h₂ = dropout(h₂)            # Apply dropout (training only)
            ...
            y = Wₙ·hₙ₋₁ + bₙ            # Output layer (no activation, no dropout)
        
        Args:
            x (torch.Tensor): Input of shape (n_samples, input_dim)
        
        Returns:
            torch.Tensor: Output predictions of shape (n_samples, output_dim)
        
        Dropout behavior:
            - During training: Randomly zeroes neurons with probability p
            - During evaluation: No dropout (model.eval() disables it)
            
            This is handled automatically by nn.Dropout:
            - training=True: Applies dropout
            - training=False: Passes input unchanged (but scales by 1/(1-p))
        
        Information Flow:
            Input → [Linear + Activation + Dropout] × N → [Linear] → Output
            
        The key insight is that composition of linear transformations
        with non-linearities enables universal function approximation.
        Dropout acts as regularization to prevent overfitting.
        """
        # Ensure input has correct shape
        if x.dim() == 1:
            x = x.unsqueeze(1)  # (n_samples,) → (n_samples, 1)
        
        # Pass through all layers
        h = x
        for i, layer in enumerate(self.layers):
            # Linear transformation
            h = layer(h)
            
            # Apply activation (except possibly on output layer)
            if i < self.num_layers - 1:
                # Always apply activation to hidden layers
                h = self.activation_fn(h)
                
                # Apply dropout to hidden layers (not to input or output)
                if self.dropout is not None:
                    h = self.dropout(h)
                    
            elif self.use_activation_output:
                # Optionally apply to output layer
                h = self.activation_fn(h)
        
        y_pred = h
        
        # For single output, flatten to (n_samples,)
        if self.layer_sizes[-1] == 1:
            y_pred = y_pred.squeeze(1)
        
        return y_pred
    
    def get_layer_outputs(self, x):
        """
        Get intermediate outputs (activations) from each layer.
        
        Useful for:
        - Visualizing what the network learns at each layer
        - Debugging gradient flow
        - Understanding representation learning
        
        Args:
            x (torch.Tensor): Input data
        
        Returns:
            list: List of tensors, one per layer (including input)
                 [input, hidden1, hidden2, ..., output]
        
        Example:
            >>> model = MultiLayerPerceptron([1, 32, 16, 1])
            >>> x = torch.randn(10, 1)
            >>> outputs = model.get_layer_outputs(x)
            >>> len(outputs)  # 4: input + 2 hidden + 1 output
            4
        """
        if x.dim() == 1:
            x = x.unsqueeze(1)
        
        layer_outputs = [x]  # Store input
        h = x
        
        for i, layer in enumerate(self.layers):
            h = layer(h)
            
            if i < self.num_layers - 1:
                h = self.activation_fn(h)
                # Note: No dropout here for analysis purposes
            elif self.use_activation_output:
                h = self.activation_fn(h)
            
            layer_outputs.append(h.detach().clone())
        
        return layer_outputs
    
    def get_architecture_summary(self):
        """
        Get a detailed summary of the network architecture.
        
        Returns:
            dict: Summary containing layer information
        """
        summary = {
            'architecture': ' → '.join(map(str, self.layer_sizes)),
            'num_layers': self.num_layers,
            'hidden_layers': self.num_layers - 1,
            'activation': self.activation_name,
            'total_params': sum(p.numel() for p in self.parameters()),
            'layer_details': []
        }
        
        for i, layer in enumerate(self.layers):
            layer_info = {
                'layer_num': i + 1,
                'input_dim': self.layer_sizes[i],
                'output_dim': self.layer_sizes[i + 1],
                'weights_shape': list(layer.weight.shape),
                'bias_shape': list(layer.bias.shape),
                'num_params': layer.weight.numel() + layer.bias.numel()
            }
            summary['layer_details'].append(layer_info)
        
        return summary
    
    def print_architecture(self):
        """Print a human-readable architecture summary."""
        summary = self.get_architecture_summary()
        
        print("="*70)
        print("MULTI-LAYER PERCEPTRON ARCHITECTURE")
        print("="*70)
        print(f"Architecture: {summary['architecture']}")
        print(f"Activation Function: {summary['activation']}")
        print(f"Total Layers: {summary['num_layers']} (including output)")
        print(f"Hidden Layers: {summary['hidden_layers']}")
        print(f"Total Parameters: {summary['total_params']}")
        print("="*70)
        print("\nLayer Details:")
        print("-"*70)
        
        for layer_info in summary['layer_details']:
            layer_type = "Hidden" if layer_info['layer_num'] < self.num_layers else "Output"
            print(f"\nLayer {layer_info['layer_num']} ({layer_type}):")
            print(f"  Input dimension:  {layer_info['input_dim']}")
            print(f"  Output dimension: {layer_info['output_dim']}")
            print(f"  Weights shape:    {layer_info['weights_shape']}")
            print(f"  Bias shape:       {layer_info['bias_shape']}")
            print(f"  Parameters:       {layer_info['num_params']}")
        
        print("="*70)
    
    def __repr__(self):
        """String representation of the MLP."""
        return (f"MultiLayerPerceptron("
                f"architecture={self.layer_sizes}, "
                f"activation='{self.activation_name}', "
                f"params={sum(p.numel() for p in self.parameters())})")


def demonstrate_universal_approximation():
    """
    Educational demonstration of the universal approximation property.
    
    Shows how increasing network capacity (width/depth) improves
    the ability to approximate complex functions.
    """
    logger.info("\n" + "="*70)
    logger.info("DEMONSTRATION: Universal Approximation Theorem")
    logger.info("="*70)
    
    # Generate a complex target function: y = sin(2πx) for x ∈ [0, 1]
    x = torch.linspace(0, 1, 100).unsqueeze(1)
    y_target = torch.sin(2 * 3.14159 * x).squeeze()
    
    # Test different architectures
    architectures = [
        [1, 1],          # Linear (no hidden layers)
        [1, 10, 1],      # Shallow with 10 units
        [1, 50, 1],      # Shallow with 50 units
        [1, 100, 1],     # Shallow with 100 units
        [1, 20, 20, 1],  # Deep with 2 hidden layers
    ]
    
    logger.info("\nTarget function: y = sin(2πx)")
    logger.info("Testing different MLP architectures...\n")
    
    for arch in architectures:
        model = MultiLayerPerceptron(arch, activation='tanh')
        
        # Random initialization approximation quality
        with torch.no_grad():
            y_pred = model(x)
            mse = torch.mean((y_pred - y_target) ** 2).item()
        
        logger.info(f"Architecture {arch}:")
        logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
        logger.info(f"  Random MSE: {mse:.6f}")
        logger.info(f"  Capacity: {'LOW' if len(arch) == 2 else 'HIGH'}")
        logger.info("")
    
    logger.info("="*70)
    logger.info("KEY INSIGHT: More parameters = Better approximation capacity!")
    logger.info("But: More parameters also = Higher risk of overfitting")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    # Demo when run directly
    from logger import configure_logging
    configure_logging()
    
    # Example 1: Single Perceptron
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 1: Single Perceptron (Linear Model)")
    logger.info("="*70)
    single = SinglePerceptron(input_dim=1, output_dim=1)
    logger.info(f"\n{single}")
    
    # Example 2: Simple MLP
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 2: Simple MLP (1 Hidden Layer)")
    logger.info("="*70)
    simple_mlp = MultiLayerPerceptron([1, 32, 1], activation='relu')
    simple_mlp.print_architecture()
    
    # Example 3: Deep MLP
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 3: Deep MLP (3 Hidden Layers)")
    logger.info("="*70)
    deep_mlp = MultiLayerPerceptron([1, 64, 32, 16, 1], activation='tanh')
    deep_mlp.print_architecture()
    
    # Example 4: Universal Approximation Demo
    demonstrate_universal_approximation()

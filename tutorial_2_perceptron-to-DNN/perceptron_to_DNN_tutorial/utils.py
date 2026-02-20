"""
utils.py
========
Utility functions for polynomial regression.

This module provides tools for feature normalization, which is critical
for numerical stability in polynomial regression.

Author: ML Tutorial Series
Target Audience: PhD students and early career postdocs
"""

import torch
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_high_order_polynomial_data(coeffs_true, poly_order, n_samples,
                                        x_min=0, x_max=10, noise_std=0.5, seed=42):
    """
    Generate synthetic data from a high-order polynomial.
    
    Creates data following the model:
        y = Σ(coeffs[i] * x^i) + noise
    
    This is useful for demonstrating:
    - Polynomial regression with neural networks
    - Feature normalization importance
    - Overfitting with high-order polynomials
    - Vanishing gradient problem in deep networks
    
    Args:
        coeffs_true (list or array): True polynomial coefficients [c0, c1, c2, ...]
                                     where y = c0 + c1*x + c2*x^2 + ...
        poly_order (int): Maximum polynomial order (degree)
        n_samples (int): Number of data points to generate
        x_min (float): Minimum x value
        x_max (float): Maximum x value
        noise_std (float): Standard deviation of Gaussian noise
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (x, y) where
            x: torch.Tensor of shape (n_samples, 1) - input features
            y: torch.Tensor of shape (n_samples, 1) - noisy target values
    
    Example:
        >>> coeffs = [1.0, 0.5, -0.1]  # y = 1.0 + 0.5*x - 0.1*x^2
        >>> x, y = generate_high_order_polynomial_data(
        ...     coeffs_true=coeffs,
        ...     poly_order=2,
        ...     n_samples=100,
        ...     x_min=0,
        ...     x_max=10,
        ...     noise_std=0.1,
        ...     seed=42
        ... )
        >>> print(f"Generated {len(x)} samples")
        >>> print(f"y range: [{y.min():.2f}, {y.max():.2f}]")
    
    Pedagogical Note:
        High-order polynomials (order > 5) are:
        - Powerful: Can fit complex patterns
        - Dangerous: Prone to overfitting
        - Unstable: Require feature normalization
        
        This function helps demonstrate these properties, especially
        the vanishing gradient problem in deep networks!
        
        For Tutorial 2, we use 9th-order polynomials to create a
        complex target function that requires deep networks to learn.
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Generate uniformly spaced x values
    x = torch.linspace(x_min, x_max, n_samples).reshape(-1, 1)
    
    # Compute polynomial: y = Σ(coeffs[i] * x^i)
    y = torch.zeros(n_samples, 1)
    
    for i in range(min(poly_order + 1, len(coeffs_true))):
        y += coeffs_true[i] * (x ** i)
    
    # Add Gaussian noise
    if noise_std > 0:
        noise = torch.randn_like(y) * noise_std
        y += noise
    
    return x, y.flatten()


def demonstrate_polynomial_data_generation():
    """
    Educational demonstration of polynomial data generation.
    
    Shows how different polynomial orders create different complexity levels.
    This is used in Tutorial 2 to demonstrate the vanishing gradient problem.
    """
    import matplotlib.pyplot as plt
    
    logger.info("\n" + "="*70)
    logger.info("DEMONSTRATION: Polynomial Data Generation")
    logger.info("="*70)
    
    # Define polynomial coefficients of increasing complexity
    coeffs_linear = [1.0, 0.5]                                    # Linear: y = 1 + 0.5x
    coeffs_quadratic = [1.0, 0.5, -0.1]                          # Quadratic
    coeffs_cubic = [1.0, 0.5, -0.1, 0.01]                        # Cubic
    coeffs_high = [1.0, 0.5, -0.1, 0.01, -0.001, 0.00005]       # 5th order
    
    configs = [
        (coeffs_linear, 1, "Linear (order 1)"),
        (coeffs_quadratic, 2, "Quadratic (order 2)"),
        (coeffs_cubic, 3, "Cubic (order 3)"),
        (coeffs_high, 5, "High-order (order 5)")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (coeffs, order, title) in enumerate(configs):
        # Generate clean data (no noise)
        x_clean, y_clean = generate_high_order_polynomial_data(
            coeffs_true=coeffs,
            poly_order=order,
            n_samples=100,
            x_min=0,
            x_max=10,
            noise_std=0.0,
            seed=42
        )
        
        # Generate noisy data
        x_noisy, y_noisy = generate_high_order_polynomial_data(
            coeffs_true=coeffs,
            poly_order=order,
            n_samples=50,
            x_min=0,
            x_max=10,
            noise_std=0.5,
            seed=42
        )
        
        # Plot
        axes[idx].plot(x_clean.numpy(), y_clean.numpy(), 'b-', linewidth=2, label='True function')
        axes[idx].scatter(x_noisy.numpy(), y_noisy.numpy(), alpha=0.5, color='red', 
                         edgecolors='black', s=50, label='Noisy samples')
        axes[idx].set_xlabel('x', fontsize=12)
        axes[idx].set_ylabel('y', fontsize=12)
        axes[idx].set_title(title, fontsize=14, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        
        logger.info(f"\n{title}:")
        logger.info(f"  Coefficients: {coeffs}")
        logger.info(f"  y range: [{y_clean.min():.2f}, {y_clean.max():.2f}]")
    
    plt.tight_layout()
    plt.savefig('polynomial_data_generation_demo.png', dpi=150, bbox_inches='tight')
    logger.info("\nVisualization saved as 'polynomial_data_generation_demo.png'")
    plt.show()
    
    logger.info("\n" + "="*70)
    logger.info("KEY INSIGHT: Higher-order polynomials create more complex patterns!")
    logger.info("            But they're harder to learn and more prone to overfitting.")
    logger.info("="*70 + "\n")



class FeatureNormalizer:
    """
    Normalizes features to a standard range for numerical stability.
    
    For polynomial regression, normalizing x BEFORE computing powers
    is crucial to prevent numerical overflow/underflow.
    
    Example:
        If x ∈ [0, 10] and we compute x⁵:
        - Without normalization: x⁵ ∈ [0, 100,000] 🚨 HUGE RANGE
        - With normalization to [-1, 1]: x⁵ ∈ [-1, 1] ✅ STABLE
    
    This is especially important for polynomial orders > 2.
    """
    
    def __init__(self, method='standardize'):
        """
        Initialize the normalizer.
        
        Args:
            method (str): Normalization method
                - 'standardize': (x - mean) / std  → mean=0, std=1
                - 'minmax': (x - min) / (max - min)  → range [0, 1]
                - 'symmetric': 2*(x - min)/(max - min) - 1  → range [-1, 1]
        """
        self.method = method
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.fitted = False
    
    def fit(self, x):
        """
        Compute normalization parameters from data.
        
        Args:
            x (torch.Tensor): Input data to fit normalization parameters
        """
        if self.method == 'standardize':
            self.mean = x.mean()
            self.std = x.std()
            if self.std == 0:
                self.std = torch.tensor(1.0)  # Avoid division by zero
                logger.warning("Standard deviation is zero, setting to 1.0")
        
        elif self.method in ['minmax', 'symmetric']:
            self.min = x.min()
            self.max = x.max()
            if self.min == self.max:
                self.max = self.min + 1.0  # Avoid division by zero
                logger.warning("Min equals max, adjusting max to min+1")
        
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        self.fitted = True
        logger.info(f"Fitted {self.method} normalizer: "
                   f"mean={self.mean}, std={self.std}, min={self.min}, max={self.max}")
    
    def transform(self, x):
        """
        Normalize data using fitted parameters.
        
        Args:
            x (torch.Tensor): Data to normalize
        
        Returns:
            torch.Tensor: Normalized data
        """
        if not self.fitted:
            raise RuntimeError("Normalizer must be fit before transform")
        
        if self.method == 'standardize':
            return (x - self.mean) / self.std
        
        elif self.method == 'minmax':
            return (x - self.min) / (self.max - self.min)
        
        elif self.method == 'symmetric':
            # Maps to [-1, 1]
            return 2 * (x - self.min) / (self.max - self.min) - 1
    
    def inverse_transform(self, x_normalized):
        """
        Denormalize data back to original scale.
        
        Args:
            x_normalized (torch.Tensor): Normalized data
        
        Returns:
            torch.Tensor: Original scale data
        """
        if not self.fitted:
            raise RuntimeError("Normalizer must be fit before inverse_transform")
        
        if self.method == 'standardize':
            return x_normalized * self.std + self.mean
        
        elif self.method == 'minmax':
            return x_normalized * (self.max - self.min) + self.min
        
        elif self.method == 'symmetric':
            # Reverse: x = ((x_norm + 1) / 2) * (max - min) + min
            return ((x_normalized + 1) / 2) * (self.max - self.min) + self.min
    
    def fit_transform(self, x):
        """
        Fit normalizer and transform data in one step.
        
        Args:
            x (torch.Tensor): Data to fit and normalize
        
        Returns:
            torch.Tensor: Normalized data
        """
        self.fit(x)
        return self.transform(x)


def demonstrate_normalization_effect():
    """
    Educational demonstration of why normalization is critical.
    
    Shows the range of polynomial features with and without normalization.
    """
    logger.info("\n" + "="*70)
    logger.info("DEMONSTRATION: Why Feature Normalization Matters")
    logger.info("="*70)
    
    # Create sample data
    x = torch.linspace(0, 10, 5)  # [0, 2.5, 5, 7.5, 10]
    
    logger.info(f"\nOriginal x: {x.tolist()}")
    logger.info(f"Range: [{x.min():.1f}, {x.max():.1f}]")
    
    # Show polynomial features WITHOUT normalization
    logger.info("\nWithout normalization:")
    for order in [1, 2, 3, 4, 5]:
        x_power = x ** order
        logger.info(f"  x^{order}: range [{x_power.min():.1f}, {x_power.max():.1f}] "
                   f"(span: {x_power.max() - x_power.min():.1f})")
    
    # Show polynomial features WITH normalization
    normalizer = FeatureNormalizer(method='symmetric')
    x_norm = normalizer.fit_transform(x)
    
    logger.info(f"\nNormalized x: {x_norm.tolist()}")
    logger.info(f"Range: [{x_norm.min():.1f}, {x_norm.max():.1f}]")
    
    logger.info("\nWith normalization to [-1, 1]:")
    for order in [1, 2, 3, 4, 5]:
        x_power = x_norm ** order
        logger.info(f"  x^{order}: range [{x_power.min():.1f}, {x_power.max():.1f}] "
                   f"(span: {x_power.max() - x_power.min():.1f})")
    
    logger.info("\n" + "="*70)
    logger.info("KEY INSIGHT: All polynomial features stay in [-1, 1] range!")
    logger.info("This prevents numerical overflow and enables stable training.")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    # Demo when run directly
    import logging
    from logger import configure_logging
    configure_logging()
    
    demonstrate_normalization_effect()


# ============================================================================
# CLASSIFICATION UTILITIES
# ============================================================================

class RegressionToClassificationConverter:
    """
    Convert a regression problem into a classification problem.
    
    This utility demonstrates that regression and classification are
    mathematically similar by discretizing continuous outputs into bins.
    
    Key Pedagogical Insight:
        - Regression: Predict y ∈ ℝ (continuous)
        - Classification: Predict y ∈ {0, 1, ..., K-1} (discrete)
        
        By discretizing regression targets into bins, we transform
        the problem into classification! The network learns the same
        underlying function, just with a different loss function.
    
    Example:
        If y ∈ [0, 10] and we use 5 bins:
        - Bin 0: y ∈ [0, 2)   → class 0
        - Bin 1: y ∈ [2, 4)   → class 1
        - Bin 2: y ∈ [4, 6)   → class 2
        - Bin 3: y ∈ [6, 8)   → class 3
        - Bin 4: y ∈ [8, 10]  → class 4
    """
    
    def __init__(self, n_bins=5, strategy='uniform'):
        """
        Initialize the converter.
        
        Args:
            n_bins (int): Number of bins/classes to create
            strategy (str): Binning strategy
                - 'uniform': Equal-width bins
                - 'quantile': Equal-frequency bins (same number of samples per bin)
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.bin_edges = None
        self.y_min = None
        self.y_max = None
        self.fitted = False
    
    def fit(self, y_continuous):
        """
        Learn bin boundaries from continuous target values.
        
        Args:
            y_continuous (torch.Tensor): Continuous target values
        """
        self.y_min = y_continuous.min().item()
        self.y_max = y_continuous.max().item()
        
        if self.strategy == 'uniform':
            # Equal-width bins
            self.bin_edges = torch.linspace(self.y_min, self.y_max, self.n_bins + 1)
        
        elif self.strategy == 'quantile':
            # Equal-frequency bins (quantiles)
            quantiles = torch.linspace(0, 1, self.n_bins + 1)
            self.bin_edges = torch.quantile(y_continuous, quantiles)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self.fitted = True
        logger.info(f"Fitted RegressionToClassificationConverter:")
        logger.info(f"  Strategy: {self.strategy}")
        logger.info(f"  Number of bins: {self.n_bins}")
        logger.info(f"  Bin edges: {self.bin_edges.tolist()}")
    
    def transform(self, y_continuous):
        """
        Convert continuous values to discrete class labels.
        
        Args:
            y_continuous (torch.Tensor): Continuous values
        
        Returns:
            torch.Tensor: Class labels (integers 0 to n_bins-1)
        """
        if not self.fitted:
            raise RuntimeError("Converter must be fit before transform")
        
        # Use torch.bucketize to assign samples to bins
        # bucketize returns bin indices (0 to n_bins)
        class_labels = torch.bucketize(y_continuous, self.bin_edges[1:-1], right=False)
        
        # Clamp to valid range [0, n_bins-1]
        class_labels = torch.clamp(class_labels, 0, self.n_bins - 1)
        
        return class_labels.long()
    
    def inverse_transform(self, class_labels):
        """
        Convert class labels back to continuous values (bin centers).
        
        Args:
            class_labels (torch.Tensor): Class labels
        
        Returns:
            torch.Tensor: Approximate continuous values (bin centers)
        """
        if not self.fitted:
            raise RuntimeError("Converter must be fit before inverse_transform")
        
        # Compute bin centers
        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        
        # Map each class label to its bin center
        y_continuous_approx = bin_centers[class_labels]
        
        return y_continuous_approx
    
    def fit_transform(self, y_continuous):
        """
        Fit and transform in one step.
        
        Args:
            y_continuous (torch.Tensor): Continuous values
        
        Returns:
            torch.Tensor: Class labels
        """
        self.fit(y_continuous)
        return self.transform(y_continuous)
    
    def get_bin_info(self):
        """
        Get information about the bins.
        
        Returns:
            dict: Dictionary with bin information
        """
        if not self.fitted:
            return None
        
        bin_info = {
            'n_bins': self.n_bins,
            'strategy': self.strategy,
            'bin_edges': self.bin_edges.tolist(),
            'bin_centers': ((self.bin_edges[:-1] + self.bin_edges[1:]) / 2).tolist(),
            'bin_widths': (self.bin_edges[1:] - self.bin_edges[:-1]).tolist()
        }
        
        return bin_info
    
    def visualize_discretization(self, y_continuous, y_classes):
        """
        Visualize how continuous values are discretized.
        
        Args:
            y_continuous (torch.Tensor): Original continuous values
            y_classes (torch.Tensor): Discretized class labels
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Continuous values
        axes[0].hist(y_continuous.numpy(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        for edge in self.bin_edges:
            axes[0].axvline(edge.item(), color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Continuous y values', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Original Continuous Distribution\n(Red lines = bin boundaries)', 
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Discretized classes
        class_counts = torch.bincount(y_classes, minlength=self.n_bins)
        axes[1].bar(range(self.n_bins), class_counts.numpy(), 
                   color='green', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Class Label', fontsize=12)
        axes[1].set_ylabel('Number of Samples', fontsize=12)
        axes[1].set_title(f'Discretized into {self.n_bins} Classes', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xticks(range(self.n_bins))
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('regression_to_classification_discretization.png', dpi=150, bbox_inches='tight')
        logger.info("Discretization visualization saved as 'regression_to_classification_discretization.png'")
        plt.show()


def demonstrate_regression_classification_equivalence():
    """
    Educational demonstration showing regression ≈ classification with fine bins.
    
    Key Insight:
        As the number of bins increases, classification converges to regression!
        
        Few bins (n=2):  Very coarse classification
        Many bins (n=100): Nearly continuous (almost regression)
        
        This shows they are the same problem, just discretized differently!
    """
    logger.info("\n" + "="*70)
    logger.info("DEMONSTRATION: Regression ≈ Classification Equivalence")
    logger.info("="*70)
    
    # Generate sample continuous data
    y_continuous = torch.linspace(0, 10, 1000) + torch.randn(1000) * 0.5
    
    logger.info("\nOriginal continuous data:")
    logger.info(f"  Range: [{y_continuous.min():.2f}, {y_continuous.max():.2f}]")
    logger.info(f"  Mean: {y_continuous.mean():.2f}")
    logger.info(f"  Std: {y_continuous.std():.2f}")
    
    # Try different numbers of bins
    for n_bins in [2, 5, 10, 20, 50, 100]:
        converter = RegressionToClassificationConverter(n_bins=n_bins, strategy='uniform')
        y_classes = converter.fit_transform(y_continuous)
        y_reconstructed = converter.inverse_transform(y_classes)
        
        # Compute reconstruction error
        reconstruction_error = torch.mean((y_continuous - y_reconstructed) ** 2)
        
        logger.info(f"\n  n_bins = {n_bins:3d}:")
        logger.info(f"    Reconstruction MSE: {reconstruction_error:.6f}")
        logger.info(f"    Information loss: {100 * reconstruction_error / y_continuous.var():.2f}%")
    
    logger.info("\n" + "="*70)
    logger.info("OBSERVATION: More bins → Lower reconstruction error")
    logger.info("             As bins → ∞, classification → regression!")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    # Demo when run directly
    import logging
    from logger import configure_logging
    configure_logging()
    
    # Original demonstration
    demonstrate_normalization_effect()
    
    # New classification demonstrations
    demonstrate_regression_classification_equivalence()


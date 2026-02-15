"""
data_generator.py
=================
Time-series polynomial data generation for Tutorial 3.

This module extends the polynomial regression problem from Tutorials 1 & 2
into a TIME-SERIES FORECASTING problem.

KEY PEDAGOGICAL CONCEPT:
------------------------
In Tutorials 1 & 2, we had STATIC polynomials:
    y = Σᵢ aᵢ xⁱ + ε
    
In Tutorial 3, we introduce TIME-VARYING coefficients:
    y(x, t) = Σᵢ aᵢ(t) xⁱ + ε
    
Where coefficients drift smoothly over time:
    aᵢ(t) = aᵢ⁽⁰⁾ + Aᵢ sin(2π fᵢ t + φᵢ)

This creates a SEQUENCE-TO-SEQUENCE problem:
    Input:  [T time steps of polynomial values]
    Output: [H future time steps to forecast]

Author: ML Tutorial Series
Target Audience: PhD students and early career postdocs
"""

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PolynomialTimeSeriesDataset:
    """
    Generate time-series data from polynomials with time-varying coefficients.
    
    This class creates data where the polynomial's coefficients drift over time,
    creating realistic dynamics similar to:
    - Economic trends with seasonal patterns
    - Temperature data with yearly cycles
    - Market behavior with slow structural changes
    
    Mathematical Formulation:
    -------------------------
    For polynomial order k, at time t:
    
        y(x, t) = Σᵢ₌₀ᵏ aᵢ(t) xⁱ + εₜ
    
    Where coefficients evolve as:
    
        aᵢ(t) = aᵢ⁽⁰⁾ + Aᵢ sin(2π fᵢ t / T + φᵢ)
        
    Parameters:
    -----------
    - aᵢ⁽⁰⁾: Base coefficient value
    - Aᵢ: Amplitude of oscillation
    - fᵢ: Frequency (cycles per T time steps)
    - φᵢ: Phase offset
    - εₜ: Gaussian noise N(0, σ²)
    
    Data Shape:
    -----------
    Generated sequences have shape [T, L] where:
    - T: Number of time steps (temporal dimension)
    - L: Number of spatial samples (x-axis samples)
    
    For batched training, reshape to [B, T, L] where B is batch size.
    """
    
    def __init__(
        self,
        coeffs_base,
        poly_order,
        T=100,
        L=50,
        x_min=-2.0,
        x_max=2.0,
        noise_std=0.1,
        amplitude_scale=0.3,
        frequencies=None,
        phases=None,
        seed=None
    ):
        """
        Initialize the time-series polynomial generator.
        
        Args:
            coeffs_base (list): Base polynomial coefficients [a₀⁽⁰⁾, a₁⁽⁰⁾, ..., aₖ⁽⁰⁾]
            poly_order (int): Polynomial order k (should match len(coeffs_base)-1)
            T (int): Number of time steps in sequence
            L (int): Number of spatial samples (x-axis points)
            x_min, x_max (float): Range for input x
            noise_std (float): Standard deviation of Gaussian noise
            amplitude_scale (float): Scale factor for coefficient oscillation amplitudes
                - Higher values = more dramatic changes over time
                - Typical range: 0.1 (subtle) to 0.5 (dramatic)
            frequencies (list, optional): Oscillation frequencies for each coefficient
                - If None, uses [1, 2, 3, ...] (increasing frequencies)
            phases (list, optional): Phase offsets for each coefficient
                - If None, uses random phases
            seed (int, optional): Random seed for reproducibility
        """
        assert len(coeffs_base) == poly_order + 1, \
            f"coeffs_base length {len(coeffs_base)} must equal poly_order+1={poly_order+1}"
        
        self.coeffs_base = np.array(coeffs_base)
        self.poly_order = poly_order
        self.T = T
        self.L = L
        self.x_min = x_min
        self.x_max = x_max
        self.noise_std = noise_std
        self.amplitude_scale = amplitude_scale
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Set up oscillation frequencies (how fast each coefficient changes)
        if frequencies is None:
            # Default: increasing frequencies for higher-order terms
            self.frequencies = np.arange(1, poly_order + 2)
        else:
            assert len(frequencies) == poly_order + 1
            self.frequencies = np.array(frequencies)
        
        # Set up phase offsets (where in the cycle each coefficient starts)
        if phases is None:
            # Random phases for variety
            self.phases = np.random.uniform(0, 2*np.pi, poly_order + 1)
        else:
            assert len(phases) == poly_order + 1
            self.phases = np.array(phases)
        
        # Compute amplitudes: proportional to base coefficient magnitude
        # (larger coefficients get larger oscillations)
        self.amplitudes = amplitude_scale * np.abs(self.coeffs_base)
        
        logger.info(f"Initialized PolynomialTimeSeriesDataset:")
        logger.info(f"  Polynomial order: {self.poly_order}")
        logger.info(f"  Time steps (T): {self.T}")
        logger.info(f"  Spatial samples (L): {self.L}")
        logger.info(f"  X range: [{self.x_min}, {self.x_max}]")
        logger.info(f"  Noise std: {self.noise_std}")
        logger.info(f"  Amplitude scale: {self.amplitude_scale}")
        logger.info(f"  Base coefficients: {self.coeffs_base}")
        logger.info(f"  Oscillation amplitudes: {self.amplitudes}")
        logger.info(f"  Frequencies: {self.frequencies}")
    
    def generate_time_varying_coefficients(self):
        """
        Generate time-varying polynomial coefficients.
        
        Returns:
            np.ndarray: Coefficient matrix of shape [T, k+1]
                - coeffs[t, i] = value of coefficient aᵢ at time t
        """
        # Time axis normalized to [0, 1]
        t_normalized = np.linspace(0, 1, self.T)
        
        # Initialize coefficient matrix
        coeffs_over_time = np.zeros((self.T, self.poly_order + 1))
        
        # Generate each coefficient's time evolution
        for i in range(self.poly_order + 1):
            # Base value + sinusoidal oscillation
            coeffs_over_time[:, i] = (
                self.coeffs_base[i] + 
                self.amplitudes[i] * np.sin(
                    2 * np.pi * self.frequencies[i] * t_normalized + self.phases[i]
                )
            )
        
        return coeffs_over_time
    
    def generate_sequence(self, return_coefficients=False):
        """
        Generate a complete time-series sequence.
        
        Returns:
            tuple: (x, y_sequence) or (x, y_sequence, coeffs_over_time)
                - x: Spatial samples, shape [L]
                - y_sequence: Polynomial values over time, shape [T, L]
                - coeffs_over_time: Time-varying coefficients, shape [T, k+1] (if requested)
        """
        # Generate spatial samples (x-axis)
        x = np.linspace(self.x_min, self.x_max, self.L)
        
        # Generate time-varying coefficients
        coeffs_over_time = self.generate_time_varying_coefficients()
        
        # Initialize output sequence
        y_sequence = np.zeros((self.T, self.L))
        
        # Generate y(x, t) for each time step
        for t in range(self.T):
            # Evaluate polynomial at time t
            y_t = np.zeros(self.L)
            for i in range(self.poly_order + 1):
                y_t += coeffs_over_time[t, i] * (x ** i)
            
            # Add noise
            noise = np.random.randn(self.L) * self.noise_std
            y_sequence[t] = y_t + noise
        
        # Convert to torch tensors
        x = torch.from_numpy(x).float()
        y_sequence = torch.from_numpy(y_sequence).float()
        
        if return_coefficients:
            coeffs_over_time = torch.from_numpy(coeffs_over_time).float()
            return x, y_sequence, coeffs_over_time
        
        return x, y_sequence
    
    def generate_batch(self, batch_size, history_length, forecast_horizon):
        """
        Generate a batch of sequence-to-sequence training examples.
        
        This creates the core training data structure for Tutorial 3:
        - Input: [B, T, L] - batch of historical sequences
        - Target: [B, H, L] - batch of future sequences to predict
        
        Args:
            batch_size (int): Number of sequences in batch
            history_length (int): Number of historical time steps (T)
            forecast_horizon (int): Number of future time steps to predict (H)
        
        Returns:
            tuple: (x, y_history, y_future)
                - x: Spatial samples, shape [L]
                - y_history: Historical data, shape [B, T, L]
                - y_future: Future data to predict, shape [B, H, L]
        """
        assert history_length + forecast_horizon <= self.T, \
            f"history_length ({history_length}) + forecast_horizon ({forecast_horizon}) " \
            f"must be <= total time steps ({self.T})"
        
        # Generate full sequence
        x, y_full_sequence = self.generate_sequence()  # [T, L]
        
        # Create batch by sliding window
        y_history_batch = []
        y_future_batch = []
        
        for _ in range(batch_size):
            # Randomly sample starting point
            max_start = self.T - history_length - forecast_horizon
            start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            
            # Extract history and future
            history = y_full_sequence[start_idx : start_idx + history_length]  # [T, L]
            future = y_full_sequence[
                start_idx + history_length : start_idx + history_length + forecast_horizon
            ]  # [H, L]
            
            y_history_batch.append(history)
            y_future_batch.append(future)
        
        # Stack into batches
        y_history = torch.stack(y_history_batch, dim=0)  # [B, T, L]
        y_future = torch.stack(y_future_batch, dim=0)    # [B, H, L]
        
        logger.debug(f"Generated batch: history {y_history.shape}, future {y_future.shape}")
        
        return x, y_history, y_future


# ============================================================================
# TUTORIAL 1 & 2 COMPATIBILITY FUNCTION
# ============================================================================

def generate_high_order_polynomial_data(
    coeffs_true,
    poly_order,
    n_samples=100,
    x_min=0,
    x_max=10,
    noise_std=0.5,
    seed=None
):
    """
    Generate STATIC polynomial data (Tutorial 1 & 2 format).
    
    This is the original polynomial generation function from Tutorials 1 & 2.
    It creates a SINGLE snapshot of the polynomial without time dynamics.
    
    Mathematical Form:
        y = Σᵢ₌₀ᵏ aᵢ xⁱ + ε,  ε ~ N(0, σ²)
    
    Args:
        coeffs_true (list): Polynomial coefficients [a₀, a₁, ..., aₖ]
        poly_order (int): Order of polynomial (should match len(coeffs_true)-1)
        n_samples (int): Number of data points to generate
        x_min, x_max (float): Range for input x
        noise_std (float): Standard deviation of Gaussian noise
        seed (int, optional): Random seed for reproducibility
    
    Returns:
        tuple: (x, y_noisy) tensors
            - x: Input values, shape [n_samples]
            - y_noisy: Noisy outputs, shape [n_samples]
    
    Pedagogical Note:
    -----------------
    This function is included for BACKWARDS COMPATIBILITY with Tutorials 1 & 2.
    Students can compare:
    
    Tutorial 1 & 2:  generate_high_order_polynomial_data()  →  Static problem
    Tutorial 3:      PolynomialTimeSeriesDataset()          →  Dynamic problem
    
    The transition from static to dynamic shows how the SAME mathematical
    structure (polynomials) can be extended to sequence modeling!
    """
    assert len(coeffs_true) == poly_order + 1, \
        f"coeffs_true length must equal poly_order + 1"
    
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Generate input samples
    x = torch.linspace(x_min, x_max, n_samples)
    
    # Evaluate polynomial
    y_true = torch.zeros(n_samples)
    for i, coeff in enumerate(coeffs_true):
        y_true += coeff * (x ** i)
    
    # Add noise
    noise = torch.randn(n_samples) * noise_std
    y_noisy = y_true + noise
    
    logger.debug(f"Generated static polynomial data: x {x.shape}, y {y_noisy.shape}")
    
    return x, y_noisy


# ============================================================================
# DEMONSTRATION FUNCTION
# ============================================================================

def demonstrate_time_varying_polynomial():
    """
    Educational demonstration of time-varying polynomials.
    
    Shows how polynomial coefficients drift over time and the resulting
    changes in the polynomial shape.
    """
    logger.info("\n" + "="*70)
    logger.info("DEMONSTRATION: Time-Varying Polynomial Coefficients")
    logger.info("="*70)
    
    # Create a simple quadratic polynomial
    coeffs_base = [1.0, 0.5, 0.2]  # y = 1 + 0.5x + 0.2x²
    
    dataset = PolynomialTimeSeriesDataset(
        coeffs_base=coeffs_base,
        poly_order=2,
        T=50,
        L=30,
        amplitude_scale=0.3,
        seed=42
    )
    
    x, y_sequence, coeffs_over_time = dataset.generate_sequence(return_coefficients=True)
    
    logger.info(f"\nGenerated time-series:")
    logger.info(f"  x shape: {x.shape}  (spatial samples)")
    logger.info(f"  y shape: {y_sequence.shape}  (time × spatial)")
    logger.info(f"  coeffs shape: {coeffs_over_time.shape}  (time × poly_order)")
    
    logger.info(f"\nCoefficient evolution:")
    for i in range(len(coeffs_base)):
        c_min = coeffs_over_time[:, i].min().item()
        c_max = coeffs_over_time[:, i].max().item()
        c_mean = coeffs_over_time[:, i].mean().item()
        logger.info(f"  a_{i}: base={coeffs_base[i]:.2f}, "
                   f"range=[{c_min:.2f}, {c_max:.2f}], mean={c_mean:.2f}")
    
    logger.info("\n" + "="*70)
    logger.info("KEY INSIGHT: Coefficients oscillate around base values,")
    logger.info("            creating smooth temporal dynamics!")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    # Demo when run directly
    from logger import configure_logging
    configure_logging()
    
    demonstrate_time_varying_polynomial()

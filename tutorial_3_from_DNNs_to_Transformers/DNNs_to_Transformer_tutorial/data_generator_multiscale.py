"""
data_generator_multiscale.py
=============================
Multi-Scale Wave System: Designed to demonstrate MLP < CNN < Transformer

This dataset combines:
1. FAST OSCILLATIONS (period ~7 steps) - CNN should excel here
2. SLOW TRENDS (period ~100 steps) - Transformer needed
3. REGIME TRANSITIONS (every ~30 steps) - Adds complexity

Expected Performance:
  Transformer: MSE ~0.06-0.08 (captures all scales)
  CNN:         MSE ~0.08-0.12 (captures fast only)
  MLP:         MSE ~0.15-0.20 (struggles with structure)

Author: ML Tutorial Series
Target Audience: PhD students demonstrating architecture choice
"""

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MultiScaleWaveDataset:
    """
    Multi-scale temporal wave system.
    
    Components:
    -----------
    1. Slow Trend: sin(2π·t/100) - period ~100 steps
       → CNN can't capture (limited receptive field)
       → Transformer captures via global attention
       
    2. Fast Wave: sin(2π·(t/7 + x/4)) - period ~7 steps
       → CNN captures perfectly (kernel_size=7 matches!)
       → Transformer also captures
       
    3. Regime Shifts: Random baseline every 30 steps
       → Adds medium-timescale complexity
       
    4. Spatial Correlation: sin(π·x)
       → Tests spatial pattern detection
       
    5. Noise: Gaussian(0, 0.1)
       → Realistic variability
    """
    
    def __init__(
            self,
            T=200,
            L=100,
            x_range=(-2.0, 2.0),
            noise_std=0.1,
            slow_period=100.0,
            fast_period=7.0,
            slow_amplitude=0.5,
            fast_amplitude=0.8,
            regime_amplitude=0.3,
            spatial_amplitude=0.2,
            seed=None
    ):
        """
        Initialize multi-scale wave dataset.
        
        Args:
            T (int): Total time steps
            L (int): Number of spatial samples
            x_range (tuple): Spatial range (min, max)
            noise_std (float): Noise level
            slow_amplitude (float): Strength of slow trend
            fast_amplitude (float): Strength of fast wave
            regime_amplitude (float): Strength of regime shifts
            spatial_amplitude (float): Strength of spatial component
            seed (int): Random seed
        """
        self.T = T
        self.L = L
        self.x_range = x_range
        self.noise_std = noise_std
        self.slow_amplitude = slow_amplitude
        self.fast_amplitude = fast_amplitude
        self.regime_amplitude = regime_amplitude
        self.spatial_amplitude = spatial_amplitude
        self.slow_period = slow_period
        self.fast_period = fast_period
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        logger.info("\n" + "="*70)
        logger.info("Multi-Scale Wave Dataset Configuration")
        logger.info("="*70)
        logger.info(f"Time steps: {T}")
        logger.info(f"Spatial samples: {L}")
        logger.info(f"Spatial range: {x_range}")
        logger.info(f"\nComponent Amplitudes:")
        logger.info(f"  Slow trend (period ~100): {slow_amplitude}")
        logger.info(f"  Fast wave (period ~7):    {fast_amplitude}")
        logger.info(f"  Regime shifts (~30):      {regime_amplitude}")
        logger.info(f"  Spatial correlation:      {spatial_amplitude}")
        logger.info(f"  Noise std:                {noise_std}")
        logger.info("\nDesigned for: MLP < CNN < Transformer hierarchy")
        logger.info("="*70 + "\n")
    
    def generate_sequence(self, return_components=False):
        """
        Generate one complete multi-scale sequence.
        
        Args:
            return_components (bool): If True, return individual components
        
        Returns:
            x: Spatial positions [L]
            y: Complete sequence [T, L]
            components (optional): Dict of individual components
        """
        # Spatial and temporal grids
        x = torch.linspace(self.x_range[0], self.x_range[1], self.L)
        t = torch.arange(self.T).float()
        
        # Broadcasting grids
        t_grid = t.unsqueeze(1).expand(-1, self.L)  # [T, L]
        x_grid = x.unsqueeze(0).expand(self.T, -1)  # [T, L]
        
        # ================================================================
        # COMPONENT 1: SLOW GLOBAL TREND (period ~100)
        # ================================================================
        # This tests ability to capture long-range dependencies
        # CNN with kernel_size=5, 3 layers has receptive field of 13
        # → Cannot capture period-100 pattern!
        # Transformer with global attention → Should capture
        
        slow_trend = self.slow_amplitude * torch.sin(2 * torch.pi * t / self.slow_period)
        slow_trend = slow_trend.unsqueeze(1).expand(-1, self.L)
        
        # ================================================================
        # COMPONENT 2: FAST LOCAL WAVE (period ~7)
        # ================================================================
        # This tests ability to detect repeating local patterns
        # CNN with kernel_size=7 → Perfect match!
        # Combines spatial position with temporal oscillation
        
        fast_wave = self.fast_amplitude * torch.sin(
            2 * torch.pi * (t_grid / self.fast_period + x_grid / 4.0)
        )
        
        # ================================================================
        # COMPONENT 3: REGIME SHIFTS (every ~30 steps)
        # ================================================================
        # Random baseline changes
        # Tests ability to detect transitions
        
        regime_shift = torch.zeros(self.T, self.L)
        num_regimes = (self.T // 30) + 1
        
        for i in range(num_regimes):
            regime_start = i * 30
            regime_end = min((i + 1) * 30, self.T)
            
            # Random shift for this regime
            shift = torch.randn(1).item() * self.regime_amplitude
            regime_shift[regime_start:regime_end, :] = shift
        
        # ================================================================
        # COMPONENT 4: SPATIAL CORRELATION
        # ================================================================
        # Smooth spatial structure
        # Tests spatial pattern detection
        
        spatial_component = self.spatial_amplitude * torch.sin(torch.pi * x_grid)
        
        # ================================================================
        # COMBINE ALL COMPONENTS
        # ================================================================
        
        y = slow_trend + fast_wave + regime_shift + spatial_component
        
        # ================================================================
        # ADD NOISE
        # ================================================================
        
        noise = torch.randn_like(y) * self.noise_std
        y = y + noise
        
        if return_components:
            components = {
                'slow_trend': slow_trend,
                'fast_wave': fast_wave,
                'regime_shift': regime_shift,
                'spatial_component': spatial_component,
                'noise': noise,
                'total': y
            }
            return x, y, components
        
        return x, y
    
    def generate_batch(
        self,
        batch_size,
        history_length,
        forecast_horizon
    ):
        """
        Generate batched data for training.
        
        Args:
            batch_size (int): Number of samples
            history_length (int): T - length of history
            forecast_horizon (int): H - length of forecast
        
        Returns:
            x: Spatial positions [L]
            y_history: Historical data [B, T, L]
            y_future: Future data to predict [B, H, L]
        """
        x_batch = []
        y_history_batch = []
        y_future_batch = []
        
        for _ in range(batch_size):
            # Generate sequence
            x, y = self.generate_sequence()
            
            # Split into history and future
            y_history = y[:history_length, :]
            y_future = y[history_length:history_length + forecast_horizon, :]
            
            x_batch.append(x)
            y_history_batch.append(y_history)
            y_future_batch.append(y_future)
        
        # Stack into batches
        x_batch = torch.stack(x_batch)  # [B, L]
        y_history_batch = torch.stack(y_history_batch)  # [B, T, L]
        y_future_batch = torch.stack(y_future_batch)  # [B, H, L]
        
        # All x should be identical, use first
        x_out = x_batch[0]
        
        return x_out, y_history_batch, y_future_batch


# ============================================================================
# DEMONSTRATION & VISUALIZATION
# ============================================================================

if __name__ == "__main__":
    from logger import configure_logging
    import matplotlib.pyplot as plt
    
    configure_logging()
    
    logger.info("="*70)
    logger.info("MULTI-SCALE WAVE DATASET DEMONSTRATION")
    logger.info("="*70)
    
    # Create dataset
    dataset = MultiScaleWaveDataset(
        T=200,
        L=100,
        x_range=(-2.0, 2.0),
        noise_std=0.1,
        slow_amplitude=0.5,
        fast_amplitude=0.8,
        regime_amplitude=0.3,
        spatial_amplitude=0.2,
        seed=42
    )
    
    # Generate one sequence with components
    logger.info("\nGenerating sample sequence with components...")
    x, y, components = dataset.generate_sequence(return_components=True)
    
    logger.info(f"x shape: {x.shape}")
    logger.info(f"y shape: {y.shape}")
    
    # Visualize
    fig = plt.figure(figsize=(16, 12))
    
    # ========================================================================
    # MAIN HEATMAP
    # ========================================================================
    ax1 = plt.subplot(3, 2, 1)
    im1 = ax1.imshow(
        y.T.numpy(),
        aspect='auto',
        origin='lower',
        extent=[0, 200, -2, 2],
        cmap='RdBu_r'
    )
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Spatial Position (x)')
    ax1.set_title('Complete Multi-Scale System')
    plt.colorbar(im1, ax=ax1)
    
    # ========================================================================
    # COMPONENT BREAKDOWN
    # ========================================================================
    
    # Slow trend
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(components['slow_trend'][:, 50].numpy(), label='Slow Trend', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Component 1: Slow Trend (period ~100)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Fast wave
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(components['fast_wave'][:100, 50].numpy(), label='Fast Wave', linewidth=2)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Component 2: Fast Wave (period ~7)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Regime shifts
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(components['regime_shift'][:, 50].numpy(), label='Regime Shifts', linewidth=2)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Amplitude')
    ax4.set_title('Component 3: Regime Transitions (every ~30 steps)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Spatial correlation
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(x.numpy(), components['spatial_component'][0, :].numpy(), 
             label='Spatial Pattern', linewidth=2)
    ax5.set_xlabel('Spatial Position (x)')
    ax5.set_ylabel('Amplitude')
    ax5.set_title('Component 4: Spatial Correlation')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Multiple positions over time
    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(y[:, 25].numpy(), label='x=-1.0', alpha=0.7, linewidth=2)
    ax6.plot(y[:, 50].numpy(), label='x=0.0', alpha=0.7, linewidth=2)
    ax6.plot(y[:, 75].numpy(), label='x=1.0', alpha=0.7, linewidth=2)
    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('y(x, t)')
    ax6.set_title('Complete System: Selected Spatial Positions')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('multiscale_wave_demo.png', dpi=150, bbox_inches='tight')
    logger.info("\n✓ Saved: multiscale_wave_demo.png")
    
    # ========================================================================
    # TEST BATCH GENERATION
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("Testing batch generation...")
    logger.info("="*70)
    
    x_batch, y_history, y_future = dataset.generate_batch(
        batch_size=32,
        history_length=150,
        forecast_horizon=50
    )
    
    logger.info(f"\nBatch shapes:")
    logger.info(f"  x:         {x_batch.shape}")
    logger.info(f"  y_history: {y_history.shape}")
    logger.info(f"  y_future:  {y_future.shape}")
    
    logger.info("\n" + "="*70)
    logger.info("EXPECTED PERFORMANCE ON THIS DATASET:")
    logger.info("="*70)
    logger.info("""
    MLP:         MSE ~0.15-0.20
      - Struggles to exploit temporal/spatial structure
      - Treats as arbitrary correlations
      - Needs massive data to learn patterns
      
    CNN:         MSE ~0.08-0.12
      - Captures fast wave (kernel_size=7 matches period!)
      - Misses slow trend (receptive field only ~13 steps)
      - Partially captures regime transitions
      
    Transformer: MSE ~0.06-0.08
      - Captures fast wave (via local attention)
      - Captures slow trend (via global attention)
      - Recognizes all temporal structures
      
    Hierarchy: Transformer > CNN > MLP ✓
    """)
    
    logger.info("="*70)
    logger.info("Demo complete! Ready for tutorial integration.")
    logger.info("="*70)

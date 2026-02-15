"""
main_multiscale_comparison_FAST_v2.py
======================================
IMPROVED VERSION with:
1. Fixed data overview plot (fast wave shows full time range)
2. Error analysis plots with confidence bands and MSE tracking

All other optimizations preserved.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

from logger import configure_logging
from data_generator_multiscale import MultiScaleWaveDataset
from MultiLayerPerceptron_AR import MultiLayerPerceptronAR
from CNN_AR_v2 import CNN_AR_v2
from Transformer_AR import TransformerAR

configure_logging()
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

T_total = 200
T_history = 150
H_forecast = 20
L_spatial = 100
x_range = (-2.0, 2.0)

batch_size = 128
n_train = 1000
n_val = 100
n_test = 100
n_epochs = 50
learning_rate = 0.001

USE_DIRECT_TRAINING = False  # Use teacher forcing (better results!)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

logger.info("\n" + "="*70)
logger.info("TUTORIAL 3C: ARCHITECTURE COMPARISON (IMPROVED)")
logger.info("="*70)
logger.info(f"Improvements:")
logger.info(f"  1. Fixed data visualization (full time range)")
logger.info(f"  2. Added error analysis plots")
logger.info(f"  3. Teacher forcing training (better predictions)")
logger.info("="*70 + "\n")

# ============================================================================
# PART 1: DATA
# ============================================================================

logger.info("="*70)
logger.info("GENERATING DATA")
logger.info("="*70 + "\n")

dataset = MultiScaleWaveDataset(
    T=T_total,
    L=L_spatial,
    x_range=x_range,
    noise_std=0.1,
    slow_amplitude=0.5,
    fast_amplitude=0.8,
    regime_amplitude=0.3,
    spatial_amplitude=0.2,
    seed=SEED
)

x_train, y_history_train, y_future_train = dataset.generate_batch(
    batch_size=n_train, history_length=T_history, forecast_horizon=H_forecast
)
x_val, y_history_val, y_future_val = dataset.generate_batch(
    batch_size=n_val, history_length=T_history, forecast_horizon=H_forecast
)
x_test, y_history_test, y_future_test = dataset.generate_batch(
    batch_size=n_test, history_length=T_history, forecast_horizon=H_forecast
)

logger.info(f"Data generated: {n_train} train, {n_val} val, {n_test} test")

# ============================================================================
# PART 2: IMPROVED DATA VISUALIZATION
# ============================================================================

logger.info("\n" + "="*70)
logger.info("VISUALIZING DATA")
logger.info("="*70 + "\n")

x_sample, y_sample, components = dataset.generate_sequence(return_components=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Full heatmap
ax = axes[0, 0]
im = ax.imshow(
    y_sample.T.numpy(),
    aspect='auto',
    origin='lower',
    extent=[0, T_total, -2, 2],
    cmap='RdBu_r'
)
ax.axvline(T_history, color='yellow', linestyle='--', linewidth=2, label='Present')
ax.set_xlabel('Time Step')
ax.set_ylabel('Spatial Position (x)')
ax.set_title('Multi-Scale Wave System')
ax.legend()
plt.colorbar(im, ax=ax)

# ⭐ FIX 1: Components - show FULL time range for both
ax = axes[0, 1]
ax.plot(components['slow_trend'][:, 50].numpy(), 
        label='Slow Trend (period ~100)', linewidth=2, color='blue')
ax.plot(components['fast_wave'][:, 50].numpy(),  # ← FIXED: Full range!
        label='Fast Wave (period ~7)', linewidth=2, alpha=0.7, color='red')
ax.set_xlabel('Time Step')
ax.set_ylabel('Amplitude')
ax.set_title('Individual Components (FULL TIME RANGE)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, T_total)

# Time series
ax = axes[1, 0]
ax.plot(y_sample[:, 25].numpy(), label='x=-1.0', alpha=0.7, linewidth=2)
ax.plot(y_sample[:, 50].numpy(), label='x=0.0', alpha=0.7, linewidth=2)
ax.plot(y_sample[:, 75].numpy(), label='x=1.0', alpha=0.7, linewidth=2)
ax.axvline(T_history, color='red', linestyle='--', linewidth=2, label='Forecast Start')
ax.set_xlabel('Time Step')
ax.set_ylabel('y(x, t)')
ax.set_title('Temporal Evolution')
ax.legend()
ax.grid(True, alpha=0.3)

# Spectrum
ax = axes[1, 1]
from scipy.fft import fft, fftfreq
signal = y_sample[:, 50].numpy()
n = len(signal)
yf = fft(signal)
xf = fftfreq(n, 1.0)[:n//2]
power = 2.0/n * np.abs(yf[0:n//2])
ax.plot(xf[1:], power[1:], linewidth=2)
ax.axvline(1/7.0, color='red', linestyle='--', label='Fast period ~7', linewidth=2)
ax.axvline(1/100.0, color='blue', linestyle='--', label='Slow period ~100', linewidth=2)
ax.set_xlabel('Frequency (1/steps)')
ax.set_ylabel('Power')
ax.set_title('Frequency Spectrum (Two Timescales)')
ax.set_xlim(0, 0.3)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_multiscale_data_overview_v2.png', dpi=150, bbox_inches='tight')
logger.info("Saved: 01_multiscale_data_overview_v2.png")

# ============================================================================
# PART 3: MODELS
# ============================================================================

logger.info("\n" + "="*70)
logger.info("BUILDING MODELS")
logger.info("="*70 + "\n")

mlp_ar_model = MultiLayerPerceptronAR(
    history_length=T_history, spatial_dim=L_spatial,
    hidden_dims=[256, 128], activation='relu', dropout=0.1
).to(device)
mlp_params = sum(p.numel() for p in mlp_ar_model.parameters())

cnn_ar_model = CNN_AR_v2(
    history_length=T_history, spatial_dim=L_spatial,
    channels=[32, 64, 32], kernel_sizes=[7, 7, 7],
    activation='relu', dropout=0.1,
    normalization='none', aggregation='attention'
).to(device)
cnn_params = sum(p.numel() for p in cnn_ar_model.parameters())

transformer_ar_model = TransformerAR(
    history_length=T_history, spatial_dim=L_spatial,
    d_model=64, n_heads=4, n_layers=2,
    dim_feedforward=128, dropout=0.1
).to(device)
transformer_params = sum(p.numel() for p in transformer_ar_model.parameters())

logger.info(f"MLP: {mlp_params:,} params")
logger.info(f"CNN: {cnn_params:,} params")
logger.info(f"Transformer: {transformer_params:,} params")

# ============================================================================
# PART 4: TRAINING
# ============================================================================

def train_autoregressive(model, optimizer, y_history, y_future, batch_size=32):
    """Teacher forcing autoregressive training."""
    model.train()
    total_loss = 0
    n_batches = len(y_history) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        batch_history = y_history[start_idx:end_idx].to(device)
        batch_future = y_future[start_idx:end_idx].to(device)
        
        optimizer.zero_grad()
        loss = 0
        
        for h in range(H_forecast):
            if h == 0:
                x_input = batch_history
            else:
                x_input = torch.cat([batch_history[:, h:, :], batch_future[:, :h, :]], dim=1)
            
            y_pred = model(x_input)
            loss = loss + nn.MSELoss()(y_pred[:, 0, :], batch_future[:, h, :])
        
        loss = loss / H_forecast
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / n_batches

def validate(model, y_history, y_future):
    """Autoregressive validation."""
    model.eval()
    with torch.no_grad():
        y_pred = model.forecast(y_history.to(device), n_steps=H_forecast)
        mse = nn.MSELoss()(y_pred, y_future.to(device)).item()
    return mse

logger.info("\n" + "="*70)
logger.info("TRAINING MODELS")
logger.info("="*70 + "\n")

models = {
    'MLP': (mlp_ar_model, optim.Adam(mlp_ar_model.parameters(), lr=learning_rate)),
    'CNN': (cnn_ar_model, optim.Adam(cnn_ar_model.parameters(), lr=learning_rate)),
    'Transformer': (transformer_ar_model, optim.Adam(transformer_ar_model.parameters(), lr=learning_rate))
}

results = {}

for name, (model, optimizer) in models.items():
    logger.info(f"\nTraining {name}...")
    logger.info("-" * 50)
    
    best_val_mse = float('inf')
    patience_counter = 0
    
    for epoch in range(n_epochs):
        train_loss = train_autoregressive(model, optimizer, y_history_train, y_future_train)
        val_mse = validate(model, y_history_val, y_future_val)
        
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 3 == 0:
            logger.info(f"Epoch {epoch:2d}/{n_epochs} | Train: {train_loss:.6f} | Val: {val_mse:.6f}")
        
        if patience_counter >= 5:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    test_mse = validate(model, y_history_test, y_future_test)
    results[name] = {'model': model, 'test_mse': test_mse}
    logger.info(f"✓ {name} Test MSE: {test_mse:.6f}")

# ============================================================================
# PART 5: RESULTS
# ============================================================================

logger.info("\n" + "="*70)
logger.info("RESULTS")
logger.info("="*70 + "\n")

sorted_results = sorted(results.items(), key=lambda x: x[1]['test_mse'])
for rank, (name, result) in enumerate(sorted_results, 1):
    logger.info(f"#{rank} {name:<12} MSE: {result['test_mse']:.6f}")

# ============================================================================
# PART 6: IMPROVED VISUALIZATION WITH ERROR ANALYSIS
# ============================================================================

logger.info("\n" + "="*70)
logger.info("GENERATING IMPROVED VISUALIZATIONS")
logger.info("="*70 + "\n")

test_idx = 0
y_history_sample = y_history_test[test_idx:test_idx+1].to(device)
y_future_sample = y_future_test[test_idx]

# Get predictions
predictions = {}
for name, result in results.items():
    model = result['model']
    model.eval()
    with torch.no_grad():
        pred = model.forecast(y_history_sample, n_steps=H_forecast).cpu()[0]
        predictions[name] = pred

# ⭐ FIX 2: Enhanced comparison with error analysis
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

pos = 50  # x=0 position

for i, (name, y_pred) in enumerate(predictions.items()):
    # Top row: Time series predictions
    ax = fig.add_subplot(gs[0, i])
    
    ax.plot(range(T_history), 
            y_history_test[test_idx, :, pos].numpy(),
            'k-', linewidth=2, label='History')
    ax.plot(range(T_history, T_history + H_forecast),
            y_future_sample[:, pos].numpy(),
            'b--', linewidth=2, label='True Future')
    ax.plot(range(T_history, T_history + H_forecast),
            y_pred[:, pos].numpy(),
            'r-', marker='o', markersize=4, linewidth=2, label='Prediction')
    
    ax.axvline(T_history, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time Step', fontsize=10)
    ax.set_ylabel('y(x=0, t)', fontsize=10)
    ax.set_title(f'{name} (MSE: {results[name]["test_mse"]:.4f})', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # ⭐ NEW: Middle row - Error over time
    ax = fig.add_subplot(gs[1, i])
    
    errors = (y_pred[:, pos] - y_future_sample[:, pos]).numpy()
    time_steps = range(H_forecast)
    
    ax.plot(time_steps, errors, 'r-', linewidth=2, label='Error')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.fill_between(time_steps, errors, 0, alpha=0.3, color='red')
    
    ax.set_xlabel('Forecast Step', fontsize=10)
    ax.set_ylabel('Prediction Error', fontsize=10)
    ax.set_title(f'{name}: Error Evolution', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # ⭐ NEW: Bottom row - Pointwise MSE with confidence
    ax = fig.add_subplot(gs[2, i])
    
    # Compute pointwise squared errors
    squared_errors = (y_pred - y_future_sample).pow(2).mean(dim=1).numpy()
    
    ax.plot(time_steps, squared_errors, 'b-', linewidth=2, marker='o', 
            markersize=6, label='MSE per step')
    
    # Add mean line
    mean_mse = squared_errors.mean()
    ax.axhline(mean_mse, color='r', linestyle='--', linewidth=2, 
               label=f'Mean MSE: {mean_mse:.4f}')
    
    # Add confidence band (std of errors)
    std_error = squared_errors.std()
    ax.fill_between(time_steps, 
                     mean_mse - std_error, 
                     mean_mse + std_error,
                     alpha=0.2, color='red', label=f'±1 std: {std_error:.4f}')
    
    ax.set_xlabel('Forecast Step', fontsize=10)
    ax.set_ylabel('Mean Squared Error', fontsize=10)
    ax.set_title(f'{name}: Error Accumulation', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

plt.suptitle('Model Comparison with Error Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('multiscale_comparison_with_errors.png', dpi=150, bbox_inches='tight')
logger.info("Saved: multiscale_comparison_with_errors.png")

# ============================================================================
# SUMMARY
# ============================================================================

logger.info("\n" + "="*70)
logger.info("TUTORIAL 3C COMPLETE!")
logger.info("="*70)
logger.info(f"""
Results:
--------
{sorted_results[0][0]}: MSE = {sorted_results[0][1]['test_mse']:.4f} (Best)
{sorted_results[1][0]}: MSE = {sorted_results[1][1]['test_mse']:.4f}
{sorted_results[2][0]}: MSE = {sorted_results[2][1]['test_mse']:.4f}

Improvements Applied:
---------------------
✓ Fixed data visualization (full time range)
✓ Added error evolution plots
✓ Added pointwise MSE with confidence bands
✓ Teacher forcing training (better results)

Key Lessons:
------------
1. CNN captures local patterns (fast wave)
2. Transformer handles all scales
3. Architecture choice matters for data structure!
""")
logger.info("="*70)

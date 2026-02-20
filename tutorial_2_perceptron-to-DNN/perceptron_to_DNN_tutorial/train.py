"""
train.py
========
Training loop implementation for PyTorch neural network models.

This module contains the training logic for fitting neural networks to data
using gradient descent optimization via backpropagation.

Key Concepts:
    - Forward pass: Compute predictions through the network
    - Loss computation: Measure prediction error
    - Backward pass: Compute gradients via automatic differentiation
    - Parameter update: Apply gradient descent step

Author: ML Tutorial Series
Target Audience: PhD students and early career postdocs
"""

import torch
import torch.optim as optim
from .loss import mean_squared_error
import logging

logger = logging.getLogger(__name__)


def train_model(model, x_train, y_train, num_epochs=1000, learning_rate=0.01, 
                print_every=100, verbose=True):
    """
    Train a PyTorch model using gradient descent and backpropagation.
    
    This function implements the standard neural network training loop:
        1. Forward pass: compute predictions
        2. Compute loss
        3. Backward pass: compute gradients via backpropagation
        4. Update parameters using optimizer
    
    Args:
        model (nn.Module): The neural network to train (MLP, Perceptron, etc.)
        x_train (torch.Tensor): Training input data
        y_train (torch.Tensor): Training target data
        num_epochs (int): Number of training iterations (passes through data)
        learning_rate (float): Step size for gradient descent
        print_every (int): Print loss every N epochs
        verbose (bool): Whether to print training progress
    
    Returns:
        tuple: (trained_model, training_history)
            - trained_model: The model after training
            - training_history: Dict containing:
                - 'loss': List of loss values at each epoch
    
    Training Algorithm (Gradient Descent with Backpropagation):
        Initialize parameters θ randomly
        
        For each epoch t = 1, ..., T:
            1. Forward pass:
               ŷ = f(x; θ)  [compute predictions]
            
            2. Compute loss:
               L(θ) = MSE(ŷ, y)  [measure error]
            
            3. Backward pass:
               ∇L(θ) = ∂L/∂θ  [compute gradients via chain rule]
            
            4. Update parameters:
               θ ← θ - α·∇L(θ)  [gradient descent step]
        
        where:
            - θ: model parameters (all weights and biases)
            - α: learning rate
            - ∇L(θ): gradient of loss w.r.t. parameters
            - f(x; θ): neural network function
    
    Backpropagation Insight:
        PyTorch's .backward() automatically computes ∇L(θ) using the chain rule.
        For a multi-layer network:
        
        Input → Layer1 → Layer2 → ... → Output → Loss
        
        Gradients flow backwards:
        
        ∂L/∂θ₁ ← ∂L/∂θ₂ ← ... ← ∂L/∂output ← Loss
        
        This is why it's called "back-propagation" - gradients propagate backward!
    """
    # Initialize optimizer (Stochastic Gradient Descent)
    # The optimizer will update model parameters based on their gradients
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Store training history
    loss_history = []
    
    # Initial logging
    if verbose:
        logger.info("=" * 70)
        logger.info("Training Neural Network")
        logger.info("=" * 70)
        logger.info(f"Model: {model.__class__.__name__}")
        if hasattr(model, 'layer_sizes'):
            logger.info(f"Architecture: {' → '.join(map(str, model.layer_sizes))}")
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Number of epochs: {num_epochs}")
        logger.info(f"Training samples: {len(x_train)}")
        logger.info("=" * 70)
    
    # Training loop
    for epoch in range(num_epochs):
        # ====================================================================
        # STEP 1: FORWARD PASS
        # ====================================================================
        # Compute predictions using current model parameters
        # This involves passing data through all layers of the network
        y_pred = model(x_train)
        
        # ====================================================================
        # STEP 2: COMPUTE LOSS
        # ====================================================================
        # Measure how far predictions are from true values
        # For regression, we use Mean Squared Error (MSE)
        loss = mean_squared_error(y_pred, y_train)
        
        # Store loss for analysis
        loss_history.append(loss.item())
        
        # ====================================================================
        # STEP 3: BACKWARD PASS (BACKPROPAGATION)
        # ====================================================================
        # Zero out gradients from previous iteration
        # PyTorch accumulates gradients by default, so we must clear them
        optimizer.zero_grad()
        
        # Compute gradients of loss w.r.t. all parameters
        # This is where the magic happens! PyTorch's autograd system
        # automatically computes ∂L/∂θ for all parameters θ using
        # the chain rule (backpropagation algorithm)
        loss.backward()
        
        # At this point, each parameter tensor has its .grad attribute filled
        # with the gradient: parameter.grad = ∂L/∂parameter
        
        # ====================================================================
        # STEP 4: UPDATE PARAMETERS (GRADIENT DESCENT STEP)
        # ====================================================================
        # Take a step in the direction of steepest descent
        # For each parameter θ: θ_new = θ_old - learning_rate * ∂L/∂θ
        optimizer.step()
        
        # ====================================================================
        # LOGGING
        # ====================================================================
        # Print progress at specified intervals
        if verbose and (epoch + 1) % print_every == 0:
            logger.info(f"Epoch [{epoch+1:4d}/{num_epochs}] | "
                       f"Loss: {loss.item():.6f}")
    
    # Final results
    if verbose:
        logger.info("=" * 70)
        logger.info("Training completed!")
        logger.info(f"Final loss: {loss_history[-1]:.6f}")
        logger.info(f"Initial loss: {loss_history[0]:.6f}")
        logger.info(f"Loss reduction: {loss_history[0] - loss_history[-1]:.6f} "
                   f"({100*(loss_history[0] - loss_history[-1])/loss_history[0]:.1f}%)")
        logger.info("=" * 70)
    
    # Prepare training history dictionary
    training_history = {
        'loss': loss_history,
    }
    
    return model, training_history


def evaluate_model(model, x_test, y_test):
    """
    Evaluate a trained model on test data.
    
    Args:
        model (nn.Module): Trained neural network
        x_test (torch.Tensor): Test input data
        y_test (torch.Tensor): Test target data
    
    Returns:
        dict: Dictionary containing evaluation metrics
            - 'mse': Mean Squared Error
            - 'rmse': Root Mean Squared Error
    
    Note:
        For neural networks, we set the model to evaluation mode
        (though this only matters for models with dropout or batch norm).
        We also disable gradient computation for efficiency.
    """
    # Set model to evaluation mode
    # (important for models with dropout, batch normalization, etc.)
    model.eval()
    
    # Disable gradient computation for efficiency
    # We don't need gradients during evaluation
    with torch.no_grad():
        y_pred = model(x_test)
        mse = mean_squared_error(y_pred, y_test)
        rmse = torch.sqrt(mse)
    
    # Set model back to training mode
    model.train()
    
    return {
        'mse': mse.item(),
        'rmse': rmse.item()
    }


def train_with_validation(model, x_train, y_train, x_val, y_val,
                         num_epochs=1000, learning_rate=0.01,
                         print_every=100, early_stopping_patience=None,
                         verbose=True):
    """
    Train a model with validation set monitoring.
    
    This extended training function:
    - Monitors both training and validation loss
    - Implements optional early stopping
    - Tracks best model based on validation loss
    
    Args:
        model (nn.Module): Neural network to train
        x_train, y_train: Training data
        x_val, y_val: Validation data
        num_epochs (int): Maximum number of epochs
        learning_rate (float): Learning rate
        print_every (int): Print frequency
        early_stopping_patience (int): Stop if val loss doesn't improve for N epochs
        verbose (bool): Print progress
    
    Returns:
        tuple: (best_model, training_history)
    
    Pedagogical Note:
        Validation monitoring is crucial for detecting overfitting:
        - Training loss keeps decreasing → model fitting training data
        - Validation loss starts increasing → model overfitting!
        
        Early stopping prevents overfitting by stopping when validation
        performance degrades.
    """
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    train_loss_history = []
    val_loss_history = []
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    
    if verbose:
        logger.info("=" * 70)
        logger.info("Training with Validation Monitoring")
        logger.info("=" * 70)
        logger.info(f"Training samples: {len(x_train)}")
        logger.info(f"Validation samples: {len(x_val)}")
        if early_stopping_patience:
            logger.info(f"Early stopping patience: {early_stopping_patience} epochs")
        logger.info("=" * 70)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        y_pred_train = model(x_train)
        train_loss = mean_squared_error(y_pred_train, y_train)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        train_loss_history.append(train_loss.item())
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            y_pred_val = model(x_val)
            val_loss = mean_squared_error(y_pred_val, y_val)
        
        val_loss_history.append(val_loss.item())
        
        # Check for best model
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping check
        if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
            if verbose:
                logger.info(f"\nEarly stopping at epoch {epoch+1}")
                logger.info(f"Validation loss hasn't improved for {early_stopping_patience} epochs")
            break
        
        # Logging
        if verbose and (epoch + 1) % print_every == 0:
            logger.info(f"Epoch [{epoch+1:4d}/{num_epochs}] | "
                       f"Train Loss: {train_loss.item():.6f} | "
                       f"Val Loss: {val_loss.item():.6f} | "
                       f"Best Val: {best_val_loss:.6f}")
    
    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    if verbose:
        logger.info("=" * 70)
        logger.info(f"Training completed at epoch {epoch+1}")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        logger.info("=" * 70)
    
    training_history = {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'best_val_loss': best_val_loss,
        'best_epoch': len(val_loss_history) - epochs_without_improvement - 1
    }
    
    return model, training_history


def compute_per_sample_gradient_distributions(model, x_train, y_train, reg_type='none',
                                              lambda_l1=0.0, lambda_l2=0.0):
    """
    Compute gradient distributions (signed values) for each individual sample.
    
    This function is used to analyze the DISTRIBUTION of gradient magnitudes
    across all training samples, preserving the sign of gradients.
    
    Args:
        model: Neural network
        x_train, y_train: Training data
        reg_type, lambda_l1, lambda_l2: Regularization parameters
    
    Returns:
        dict: Maps layer_name -> list of gradient arrays (one array per sample)
              Example: {'layers.0.weight': [array1, array2, ...]}
              where each array contains all signed gradient values for that sample
    
    Pedagogical Note:
        Normal training computes gradients for ALL samples at once (batch),
        giving one aggregated gradient per parameter.
        
        To see the DISTRIBUTION of gradients with signs, we compute gradients 
        for each sample individually and store ALL gradient values (flattened).
        
        This reveals:
        - Full distribution shape (not just magnitude)
        - Positive vs negative gradient balance
        - Whether gradients are vanishing (concentrated near zero)
        - Bimodal or skewed distributions
    
    Warning: This is computationally expensive (n_samples forward/backward passes)
    and creates large data structures (all gradient values stored).
    Should only be called at specific epochs for visualization.
    """
    from .loss import mean_squared_error, compute_total_loss_with_regularization
    
    model.train()
    n_samples = len(x_train)
    
    # Initialize storage: layer_name -> list of gradient arrays
    per_sample_grad_distributions = {}
    layer_names = [name for name, _ in model.named_parameters() if 'weight' in name]
    for name in layer_names:
        per_sample_grad_distributions[name] = []
    
    # Compute gradients for each sample individually
    for i in range(n_samples):
        # Get single sample
        x_sample = x_train[i:i+1]
        y_sample = y_train[i:i+1]
        
        # Forward pass
        y_pred = model(x_sample)
        
        # Compute loss
        data_loss = mean_squared_error(y_pred, y_sample)
        total_loss, _ = compute_total_loss_with_regularization(
            data_loss, model, reg_type=reg_type,
            lambda_l1=lambda_l1, lambda_l2=lambda_l2
        )
        
        # Backward pass
        model.zero_grad()
        total_loss.backward()
        
        # Store flattened gradient values (with signs) for this sample
        for name, param in model.named_parameters():
            if 'weight' in name and param.grad is not None:
                # Flatten gradient tensor and convert to numpy
                grad_values = param.grad.data.view(-1).cpu().clone().numpy()
                per_sample_grad_distributions[name].append(grad_values)
    
    return per_sample_grad_distributions


def train_model_with_gradient_tracking(model, x_train, y_train,
                                       x_valid=None, y_valid=None,
                                       num_epochs=1000, learning_rate=0.01,
                                       reg_type='none', lambda_l1=0.0, lambda_l2=0.0,
                                       print_every=100, verbose=True,
                                       track_gradients=True,
                                       track_per_sample_gradients=False):
    """
    Train a model while tracking gradient magnitudes per layer.
    
    This function extends the standard training loop to monitor:
    - Gradient magnitudes at each layer
    - Detection of vanishing/exploding gradients
    - Effect of different activation functions on gradient flow
    - Optional: Validation loss for overfitting detection
    - Optional: Per-sample gradient distributions for histogram plotting
    
    Args:
        model (nn.Module): Neural network to train
        x_train, y_train: Training data
        x_valid, y_valid: Optional validation data (if provided, tracks validation loss)
        num_epochs (int): Number of training iterations
        learning_rate (float): Learning rate
        reg_type (str): Regularization type ('none', 'l1', 'l2', 'elastic')
        lambda_l1, lambda_l2 (float): Regularization strengths
        print_every (int): Logging frequency
        verbose (bool): Print training progress
        track_gradients (bool): Track gradient statistics (adds overhead)
        track_per_sample_gradients (bool): Track per-sample gradient distributions
                                           at first, middle, and last epochs (expensive!)
    
    Returns:
        tuple: (trained_model, training_history)
            training_history contains:
            - 'train_loss': Training data loss at each epoch
            - 'valid_loss': Validation loss at each epoch (if validation data provided)
            - 'total_loss': Total loss (data + regularization)
            - 'reg_penalty': Regularization penalty
            - 'gradient_norms': Gradient norm per layer per epoch (if track_gradients=True)
            - 'per_sample_gradient_distributions': Per-sample gradient norms at
                                                   first, middle, last epochs
                                                   (if track_per_sample_gradients=True)
                                                   Format: {epoch_idx: {layer_name: [list of norms]}}
            
            Note: For backwards compatibility, 'loss' key is an alias for 'train_loss'
    
    Pedagogical Note - Vanishing Gradient Problem:
        In deep networks with certain activation functions (sigmoid, tanh),
        gradients can become exponentially small as they backpropagate
        through layers. This happens because:
        
        1. Chain rule multiplies gradients:
           ∂L/∂W₁ = ∂L/∂h₃ · ∂h₃/∂h₂ · ∂h₂/∂h₁ · ∂h₁/∂W₁
        
        2. If each factor < 1, product shrinks exponentially:
           - tanh'(x) ∈ (0, 1]  → gradients can vanish
           - sigmoid'(x) ∈ (0, 0.25] → gradients vanish faster!
           - ReLU'(x) = 1 (for x>0) → no vanishing!
        
        This function tracks these gradients to visualize the problem.
        
        With validation data, also tracks overfitting:
        - Train loss decreases but validation loss increases → Overfitting!
        - Both decrease together → Good generalization
    """
    from .loss import mean_squared_error, compute_total_loss_with_regularization
    import torch.optim as optim
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Check if validation data provided
    has_validation = (x_valid is not None and y_valid is not None)
    
    # Determine epochs for per-sample gradient tracking
    epoch_indices_for_distributions = []
    if track_per_sample_gradients:
        epoch_indices_for_distributions = [0, num_epochs // 2, num_epochs - 1]
        logger.info(f"Per-sample gradient tracking ENABLED at epochs: {epoch_indices_for_distributions}")
    
    # Training history
    train_loss_history = []
    valid_loss_history = []
    total_loss_history = []
    reg_penalty_history = []
    gradient_norms_history = []  # List of dicts: {layer_name: norm}
    per_sample_gradient_distributions = {}  # {epoch_idx: {layer_name: [list of norms]}}
    
    if verbose:
        logger.info("=" * 70)
        logger.info("Training with Gradient Tracking" + 
                   (" + Validation" if has_validation else ""))
        logger.info("=" * 70)
        logger.info(f"Model: {model.__class__.__name__}")
        if hasattr(model, 'layer_sizes'):
            logger.info(f"Architecture: {' → '.join(map(str, model.layer_sizes))}")
            logger.info(f"Activation: {model.activation_name}")
            if hasattr(model, 'dropout_rate') and model.dropout_rate > 0:
                logger.info(f"Dropout: {model.dropout_rate}")
        logger.info(f"Regularization: {reg_type} (λ₁={lambda_l1}, λ₂={lambda_l2})")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Training samples: {len(x_train)}")
        if has_validation:
            logger.info(f"Validation samples: {len(x_valid)}")
        logger.info("=" * 70)
    
    for epoch in range(num_epochs):
        # ============================================================
        # TRAINING STEP
        # ============================================================
        model.train()  # Enable dropout if present
        
        # Forward pass
        y_pred = model(x_train)
        
        # Compute data loss
        data_loss = mean_squared_error(y_pred, y_train)
        
        # Compute total loss with regularization
        total_loss, reg_penalty = compute_total_loss_with_regularization(
            data_loss, model, reg_type=reg_type, 
            lambda_l1=lambda_l1, lambda_l2=lambda_l2
        )
        
        # Store training loss
        train_loss_history.append(data_loss.item())
        total_loss_history.append(total_loss.item())
        reg_penalty_history.append(reg_penalty.item())
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Track gradients BEFORE optimizer step
        if track_gradients:
            gradient_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Compute L2 norm of gradients
                    grad_norm = param.grad.norm().item()
                    gradient_norms[name] = grad_norm
            gradient_norms_history.append(gradient_norms)
        
        # Track per-sample gradient distributions at specific epochs
        if track_per_sample_gradients and epoch in epoch_indices_for_distributions:
            if verbose:
                logger.info(f"  [Epoch {epoch+1}] Computing per-sample gradient distributions...")
            
            # Save current model state
            saved_state = {name: param.clone() for name, param in model.named_parameters()}
            
            # Compute per-sample gradients (expensive!)
            per_sample_grads = compute_per_sample_gradient_distributions(
                model, x_train, y_train,
                reg_type=reg_type,
                lambda_l1=lambda_l1,
                lambda_l2=lambda_l2
            )
            
            # Store in history
            per_sample_gradient_distributions[epoch] = per_sample_grads
            
            # Restore model state (per-sample computation may have modified it)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(saved_state[name])
            
            if verbose:
                logger.info(f"  [Epoch {epoch+1}] Per-sample gradients collected for {len(per_sample_grads)} layers")
        
        # Update parameters
        optimizer.step()
        
        # ============================================================
        # VALIDATION STEP (if validation data provided)
        # ============================================================
        if has_validation:
            valid_loss = compute_validation_loss(
                model, x_valid, y_valid,
                reg_type=reg_type,
                lambda_l1=lambda_l1,
                lambda_l2=lambda_l2
            )
            valid_loss_history.append(valid_loss)
        
        # ============================================================
        # LOGGING
        # ============================================================
        if verbose and (epoch + 1) % print_every == 0:
            log_msg = (f"Epoch [{epoch+1:4d}/{num_epochs}] | "
                      f"Train Loss: {data_loss.item():.6f}")
            
            if has_validation:
                overfitting_gap = valid_loss - data_loss.item()
                log_msg += f" | Valid Loss: {valid_loss:.6f} | Gap: {overfitting_gap:+.6f}"
            else:
                log_msg += f" | Reg: {reg_penalty.item():.6f} | Total: {total_loss.item():.6f}"
            
            logger.info(log_msg)
    
    if verbose:
        logger.info("=" * 70)
        logger.info("Training completed!")
        logger.info(f"Final train loss: {train_loss_history[-1]:.6f}")
        
        if has_validation:
            logger.info(f"Final valid loss: {valid_loss_history[-1]:.6f}")
            final_gap = valid_loss_history[-1] - train_loss_history[-1]
            logger.info(f"Final gap (valid - train): {final_gap:+.6f}")
        else:
            logger.info(f"Final total loss: {total_loss_history[-1]:.6f}")
        
        if track_gradients:
            logger.info("\nGradient statistics (last epoch):")
            for name, norm in gradient_norms_history[-1].items():
                logger.info(f"  {name}: {norm:.6f}")
        logger.info("=" * 70)
    
    # Build training history
    training_history = {
        'train_loss': train_loss_history,
        'loss': train_loss_history,  # Backwards compatibility
        'total_loss': total_loss_history,
        'reg_penalty': reg_penalty_history,
    }
    
    # Add validation loss if available
    if has_validation:
        training_history['valid_loss'] = valid_loss_history
    
    # Add gradient norms if tracked
    if track_gradients:
        training_history['gradient_norms'] = gradient_norms_history
    
    # Add per-sample gradient distributions if tracked
    if track_per_sample_gradients:
        training_history['per_sample_gradient_distributions'] = per_sample_gradient_distributions
        if verbose:
            logger.info(f"Added per-sample gradients to history: {len(per_sample_gradient_distributions)} epochs")
    
    return model, training_history


def compute_validation_loss(model, x_valid, y_valid, reg_type='none', 
                            lambda_l1=0.0, lambda_l2=0.0):
    """
    Compute validation loss (no gradient computation, no parameter updates).
    
    This function evaluates the model on validation data to:
    - Monitor overfitting (train loss << validation loss)
    - Select best hyperparameters
    - Decide when to stop training (early stopping)
    
    CRITICAL: Returns DATA LOSS ONLY (no regularization penalty)
    The regularization penalty is the same regardless of which dataset
    is being evaluated, so including it would artificially inflate
    validation loss and give misleading overfitting measurements.
    
    Args:
        model (nn.Module): Neural network to evaluate
        x_valid, y_valid: Validation data
        reg_type (str): Regularization type (NOT used for validation loss)
        lambda_l1, lambda_l2 (float): Regularization strengths (NOT used)
    
    Returns:
        float: Validation DATA loss (MSE) - NO regularization penalty
    
    Pedagogical Note - Why Data Loss Only?
        Validation loss measures generalization:
        - How well does the model predict UNSEEN data?
        - Regularization penalty depends on weights, not data
        - Including penalty in validation loss would give misleading comparison
        
        Example (with L2 penalty = 0.8):
        WRONG:  Train = 0.5 (data), Valid = 0.5 + 0.8 = 1.3 → Gap = 0.8 (misleading!)
        RIGHT:  Train = 0.5 (data), Valid = 0.5 (data) → Gap = 0.0 (accurate!)
        
        For fair comparison: Both train and valid should measure DATA loss only.
    """
    from .loss import mean_squared_error
    
    # Set model to evaluation mode (disables dropout)
    model.eval()
    
    with torch.no_grad():  # Don't compute gradients
        # Forward pass
        y_pred = model(x_valid)
        
        # Compute DATA loss only (no regularization)
        data_loss = mean_squared_error(y_pred, y_valid)
    
    # Set model back to training mode
    model.train()
    
    # Return DATA LOSS ONLY - do NOT include regularization penalty
    return data_loss.item()


def train_model_with_validation_tracking(model, x_train, y_train,
                                         x_valid, y_valid,
                                         num_epochs=1000, learning_rate=0.01,
                                         reg_type='none', lambda_l1=0.0, lambda_l2=0.0,
                                         print_every=100, verbose=True,
                                         track_gradients=False):
    """
    Train a model while tracking both training and validation loss.
    
    This function extends training to monitor overfitting:
    - Training loss: How well model fits training data
    - Validation loss: How well model generalizes to unseen data
    - Gap between them: Measure of overfitting
    
    Args:
        model (nn.Module): Neural network to train
        x_train, y_train: Training data
        x_valid, y_valid: Validation data
        num_epochs (int): Number of training iterations
        learning_rate (float): Learning rate
        reg_type (str): Regularization type ('none', 'l1', 'l2', 'elastic')
        lambda_l1, lambda_l2 (float): Regularization strengths
        print_every (int): Logging frequency
        verbose (bool): Print training progress
        track_gradients (bool): Track gradient statistics
    
    Returns:
        tuple: (trained_model, training_history)
            training_history contains:
            - 'train_loss': Training loss at each epoch
            - 'valid_loss': Validation loss at each epoch
            - 'total_loss': Total loss (data + regularization)
            - 'reg_penalty': Regularization penalty
            - 'gradient_norms': Gradient norm per layer per epoch (if tracked)
    
    Pedagogical Note - Detecting Overfitting:
        Plot train_loss vs valid_loss:
        
        NO OVERFITTING (regularized):
        Loss
         |  Train ─────────────
         |  Valid ─────────────
         |________________________ Epochs
             Both decrease together!
        
        OVERFITTING (no regularization):
        Loss
         |  Valid ──────────/
         |  Train ──────────\
         |________________________ Epochs
             Train decreases, Valid increases!
    """
    from .loss import mean_squared_error, compute_total_loss_with_regularization
    import torch.optim as optim
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Check if validation data provided
    has_validation = (x_valid is not None and y_valid is not None)
    
    # Training history
    train_loss_history = []
    valid_loss_history = []
    total_loss_history = []
    reg_penalty_history = []
    gradient_norms_history = []
    
    if verbose:
        logger.info("=" * 70)
        logger.info("Training with Validation Tracking")
        logger.info("=" * 70)
        logger.info(f"Model: {model.__class__.__name__}")
        if hasattr(model, 'layer_sizes'):
            logger.info(f"Architecture: {' → '.join(map(str, model.layer_sizes))}")
            logger.info(f"Activation: {model.activation_name}")
            if hasattr(model, 'dropout_rate') and model.dropout_rate > 0:
                logger.info(f"Dropout: {model.dropout_rate}")
        logger.info(f"Regularization: {reg_type} (λ₁={lambda_l1}, λ₂={lambda_l2})")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Training samples: {len(x_train)}")
        logger.info(f"Validation samples: {len(x_valid)}")
        logger.info("=" * 70)
    
    for epoch in range(num_epochs):
        # ============================================================
        # TRAINING STEP
        # ============================================================
        model.train()  # Enable dropout
        
        # Forward pass
        y_pred = model(x_train)
        
        # Compute data loss
        data_loss = mean_squared_error(y_pred, y_train)
        
        # Compute total loss with regularization
        total_loss, reg_penalty = compute_total_loss_with_regularization(
            data_loss, model, reg_type=reg_type, 
            lambda_l1=lambda_l1, lambda_l2=lambda_l2
        )
        
        # Store losses
        train_loss_history.append(data_loss.item())
        total_loss_history.append(total_loss.item())
        reg_penalty_history.append(reg_penalty.item())
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Track gradients BEFORE optimizer step
        if track_gradients:
            gradient_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    gradient_norms[name] = grad_norm
            gradient_norms_history.append(gradient_norms)
        
        # Update parameters
        optimizer.step()
        
        # ============================================================
        # VALIDATION STEP (no gradient computation)
        # ============================================================
        valid_loss = compute_validation_loss(
            model, x_valid, y_valid,
            reg_type=reg_type,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2
        )
        valid_loss_history.append(valid_loss)
        
        # ============================================================
        # LOGGING
        # ============================================================
        if verbose and (epoch + 1) % print_every == 0:
            overfitting_gap = valid_loss - data_loss.item()
            logger.info(f"Epoch [{epoch+1:4d}/{num_epochs}] | "
                       f"Train Loss: {data_loss.item():.6f} | "
                       f"Valid Loss: {valid_loss:.6f} | "
                       f"Gap: {overfitting_gap:+.6f}")
    
    if verbose:
        logger.info("=" * 70)
        logger.info("Training completed!")
        logger.info(f"Final train loss: {train_loss_history[-1]:.6f}")
        logger.info(f"Final valid loss: {valid_loss_history[-1]:.6f}")
        
        # Analyze overfitting
        final_gap = valid_loss_history[-1] - train_loss_history[-1]
        logger.info(f"Final gap (valid - train): {final_gap:+.6f}")
        
        if final_gap < 0.1:
            logger.info("✓ No overfitting detected! Good generalization.")
        elif final_gap < 0.5:
            logger.info("⚠ Slight overfitting. Consider more regularization.")
        else:
            logger.info("⚠️ Significant overfitting! Increase regularization.")
        
        if track_gradients:
            logger.info("\nGradient statistics (last epoch):")
            for name, norm in gradient_norms_history[-1].items():
                logger.info(f"  {name}: {norm:.6f}")
        logger.info("=" * 70)
    
    # Build training history
    training_history = {
        'train_loss': train_loss_history,
        'loss': train_loss_history,  # Backwards compatibility
        'total_loss': total_loss_history,
        'reg_penalty': reg_penalty_history,
    }
    
    # Add validation loss if available
    if has_validation:
        training_history['valid_loss'] = valid_loss_history
    
    # Add gradient norms if tracked
    if track_gradients:
        training_history['gradient_norms'] = gradient_norms_history
    
    
    return model, training_history

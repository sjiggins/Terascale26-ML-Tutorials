"""
loss.py
=======
Loss functions for training machine learning models.

This module implements loss functions for both regression and classification tasks,
demonstrating their mathematical similarity.

Key Insight:
    Regression and classification are fundamentally similar!
    - Regression: Predict continuous values
    - Classification: Predict discrete classes
    Both learn p(y|x), just with different output spaces and loss functions.

Author: ML Tutorial Series
Target Audience: PhD students and early career postdocs
"""

import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# REGRESSION LOSSES
# ============================================================================

def mean_squared_error(y_pred, y_true):
    """
    Compute the Mean Squared Error (MSE) loss.
    
    MSE is defined as:
        MSE = (1/n) * Σ(y_pred - y_true)²
    
    This is the most common loss function for regression problems.
    It penalizes larger errors more heavily due to the squaring operation.
    
    Args:
        y_pred (torch.Tensor): Predicted values from the model
        y_true (torch.Tensor): True/observed values
    
    Returns:
        torch.Tensor: Scalar tensor containing the MSE loss
    
    Mathematical Properties:
        - Always non-negative
        - Convex (has a single global minimum)
        - Differentiable everywhere
        - Sensitive to outliers due to squaring
    
    Connection to Classification:
        MSE measures L2 distance between prediction and target.
        Cross-entropy (classification loss) measures KL divergence.
        Both measure "distance" but in different spaces!
    """
    # Ensure both tensors have the same shape
    assert y_pred.shape == y_true.shape, \
        f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}"
    
    # Compute squared differences
    squared_diff = (y_pred - y_true) ** 2
    
    # Take the mean over all samples
    mse = torch.mean(squared_diff)
    
    return mse


def root_mean_squared_error(y_pred, y_true):
    """
    Compute the Root Mean Squared Error (RMSE).
    
    RMSE = sqrt(MSE)
    
    RMSE is in the same units as the target variable, making it
    more interpretable than MSE.
    
    Args:
        y_pred (torch.Tensor): Predicted values
        y_true (torch.Tensor): True values
    
    Returns:
        torch.Tensor: Scalar tensor containing the RMSE
    """
    mse = mean_squared_error(y_pred, y_true)
    rmse = torch.sqrt(mse)
    return rmse


def mean_absolute_error(y_pred, y_true):
    """
    Compute the Mean Absolute Error (MAE).
    
    MAE = (1/n) * Σ|y_pred - y_true|
    
    MAE is less sensitive to outliers compared to MSE.
    
    Args:
        y_pred (torch.Tensor): Predicted values
        y_true (torch.Tensor): True values
    
    Returns:
        torch.Tensor: Scalar tensor containing the MAE
    """
    absolute_diff = torch.abs(y_pred - y_true)
    mae = torch.mean(absolute_diff)
    return mae


# ============================================================================
# CLASSIFICATION LOSSES
# ============================================================================

def binary_cross_entropy(y_pred, y_true):
    """
    Compute Binary Cross-Entropy loss for binary classification.
    
    BCE is defined as:
        BCE = -(1/n) * Σ[y_true * log(y_pred) + (1-y_true) * log(1-y_pred)]
    
    This is the standard loss for binary classification (2 classes).
    
    Args:
        y_pred (torch.Tensor): Predicted probabilities in [0, 1]
                              (output of sigmoid activation)
        y_true (torch.Tensor): True labels (0 or 1)
    
    Returns:
        torch.Tensor: Scalar tensor containing the BCE loss
    
    Mathematical Interpretation:
        - Measures the KL divergence between true and predicted distributions
        - Equivalent to maximum likelihood estimation
        - Convex in the logits (pre-sigmoid values)
    
    Connection to Regression:
        - MSE: Assumes Gaussian noise, minimizes L2 distance
        - BCE: Assumes Bernoulli distribution, minimizes KL divergence
        Both are maximum likelihood estimators under different assumptions!
    
    Pedagogical Note:
        BCE for 2 classes is mathematically equivalent to categorical
        cross-entropy with 2 classes. This shows the unity of classification!
    """
    # Clamp predictions to avoid log(0)
    eps = 1e-7
    y_pred = torch.clamp(y_pred, eps, 1 - eps)
    
    # Compute binary cross-entropy
    bce = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    
    return torch.mean(bce)


def categorical_cross_entropy(y_pred, y_true):
    """
    Compute Categorical Cross-Entropy loss for multi-class classification.
    
    CCE is defined as:
        CCE = -(1/n) * Σᵢ Σⱼ y_true[i,j] * log(y_pred[i,j])
    
    where j indexes over classes.
    
    Args:
        y_pred (torch.Tensor): Predicted class probabilities, shape (n_samples, n_classes)
                              (output of softmax activation)
        y_true (torch.Tensor): True labels, either:
                              - One-hot encoded: shape (n_samples, n_classes)
                              - Class indices: shape (n_samples,)
    
    Returns:
        torch.Tensor: Scalar tensor containing the CCE loss
    
    Mathematical Properties:
        - Measures KL divergence: D_KL(y_true || y_pred)
        - Convex in the logits (pre-softmax values)
        - Reduces to binary cross-entropy when n_classes = 2
    
    Key Insight:
        This is the generalization of binary cross-entropy to K classes.
        The mathematics is identical, just extended to multiple classes!
    """
    # Handle two formats: one-hot encoded or class indices
    if y_true.dim() == 1 or y_true.shape[-1] == 1:
        # Class indices format
        if y_true.dim() == 2:
            y_true = y_true.squeeze(1)
        # Use PyTorch's built-in cross-entropy (numerically stable)
        # Note: This expects logits, not probabilities
        # But we'll work with probabilities for pedagogical clarity
        eps = 1e-7
        y_pred = torch.clamp(y_pred, eps, 1 - eps)
        
        # Convert to one-hot
        n_classes = y_pred.shape[1]
        y_true_one_hot = F.one_hot(y_true.long(), num_classes=n_classes).float()
        
        # Compute cross-entropy
        cce = -torch.sum(y_true_one_hot * torch.log(y_pred), dim=1)
    else:
        # One-hot encoded format
        eps = 1e-7
        y_pred = torch.clamp(y_pred, eps, 1 - eps)
        
        # Compute cross-entropy
        cce = -torch.sum(y_true * torch.log(y_pred), dim=1)
    
    return torch.mean(cce)


def cross_entropy_from_logits(logits, y_true):
    """
    Compute cross-entropy directly from logits (more numerically stable).
    
    This is the preferred way to compute cross-entropy in practice.
    
    Args:
        logits (torch.Tensor): Raw network outputs (before softmax)
        y_true (torch.Tensor): True class labels (indices)
    
    Returns:
        torch.Tensor: Scalar tensor containing the cross-entropy loss
    
    Why use logits instead of probabilities?
        The softmax + log operations can be combined for numerical stability:
        
        Naive:     loss = -log(softmax(logits))
        Stable:    loss = -log_softmax(logits)  # Combines operations
        
        This prevents numerical underflow when probabilities are very small.
    """
    return F.cross_entropy(logits, y_true.long())


# ============================================================================
# PEDAGOGICAL COMPARISON FUNCTIONS
# ============================================================================

def compare_losses_mathematically():
    """
    Educational demonstration of the similarity between regression and
    classification losses.
    
    Key Insight:
        - Regression (MSE): Assumes y ~ N(μ, σ²), minimizes L2 distance
        - Classification (CE): Assumes y ~ Categorical(p), minimizes KL divergence
        
        Both are maximum likelihood estimators (MLE) under different assumptions!
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*70)
    logger.info("MATHEMATICAL COMPARISON: Regression vs Classification")
    logger.info("="*70)
    
    logger.info("""
    REGRESSION (MSE Loss):
    ----------------------
    Assumption: y ~ N(f(x), σ²)  [Gaussian distribution]
    
    Likelihood:
        L(θ) = ∏ᵢ (1/√(2πσ²)) * exp(-(yᵢ - f(xᵢ; θ))² / 2σ²)
    
    Negative Log-Likelihood:
        -log L(θ) = Σᵢ (yᵢ - f(xᵢ; θ))² / 2σ² + const
    
    Minimize NLL ⟹ Minimize MSE = Σᵢ (yᵢ - f(xᵢ; θ))²
    
    
    CLASSIFICATION (Cross-Entropy Loss):
    ------------------------------------
    Assumption: y ~ Categorical(p)  [Categorical distribution]
    
    Likelihood:
        L(θ) = ∏ᵢ ∏ⱼ p(yᵢ=j | xᵢ; θ)^{𝕀[yᵢ=j]}
    
    Negative Log-Likelihood:
        -log L(θ) = -Σᵢ Σⱼ 𝕀[yᵢ=j] * log p(yᵢ=j | xᵢ; θ)
    
    This IS the cross-entropy loss!
    
    
    UNIFIED VIEW:
    -------------
    Both regression and classification are maximum likelihood estimation!
    
    Regression: MLE under Gaussian assumption → MSE loss
    Classification: MLE under Categorical assumption → Cross-entropy loss
    
    The mathematics is the same, only the assumed distribution differs!
    """)
    
    logger.info("="*70)
    logger.info("This is why we can solve the SAME problem (polynomial fitting)")
    logger.info("using BOTH regression and classification approaches!")
    logger.info("="*70 + "\n")


def demonstrate_loss_behavior():
    """
    Demonstrate how different losses behave with sample predictions.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*70)
    logger.info("LOSS FUNCTION BEHAVIOR COMPARISON")
    logger.info("="*70)
    
    # Regression example
    y_true_reg = torch.tensor([1.0, 2.0, 3.0])
    y_pred_reg = torch.tensor([1.1, 2.3, 2.8])
    
    mse = mean_squared_error(y_pred_reg, y_true_reg)
    mae = mean_absolute_error(y_pred_reg, y_true_reg)
    
    logger.info("\nREGRESSION Example:")
    logger.info(f"True values:      {y_true_reg.tolist()}")
    logger.info(f"Predicted values: {y_pred_reg.tolist()}")
    logger.info(f"MSE Loss:         {mse.item():.6f}")
    logger.info(f"MAE Loss:         {mae.item():.6f}")
    
    # Binary classification example
    y_true_bin = torch.tensor([1.0, 0.0, 1.0])
    y_pred_bin = torch.tensor([0.9, 0.1, 0.8])
    
    bce = binary_cross_entropy(y_pred_bin, y_true_bin)
    
    logger.info("\nBINARY CLASSIFICATION Example:")
    logger.info(f"True labels:      {y_true_bin.tolist()}")
    logger.info(f"Predicted probs:  {y_pred_bin.tolist()}")
    logger.info(f"BCE Loss:         {bce.item():.6f}")
    
    # Multi-class classification example
    y_true_multi = torch.tensor([0, 1, 2])  # Class indices
    y_pred_multi = torch.tensor([
        [0.7, 0.2, 0.1],  # Predicted probs for sample 1
        [0.1, 0.8, 0.1],  # Predicted probs for sample 2
        [0.1, 0.1, 0.8],  # Predicted probs for sample 3
    ])
    
    cce = categorical_cross_entropy(y_pred_multi, y_true_multi)
    
    logger.info("\nMULTI-CLASS CLASSIFICATION Example:")
    logger.info(f"True classes:     {y_true_multi.tolist()}")
    logger.info(f"Predicted probs:")
    for i, probs in enumerate(y_pred_multi):
        logger.info(f"  Sample {i+1}: {probs.tolist()}")
    logger.info(f"CCE Loss:         {cce.item():.6f}")
    
    logger.info("\n" + "="*70)
    logger.info("OBSERVATION: All losses are differentiable and decrease")
    logger.info("as predictions get closer to true values!")
    logger.info("="*70 + "\n")


# ============================================================================
# REGULARIZATION LOSSES
# ============================================================================

def l1_regularization(model, lambda_l1=0.01):
    """
    Compute L1 (Lasso) regularization penalty.
    
    L1 = λ * Σ|wᵢ|
    
    L1 regularization promotes SPARSITY - it drives some weights to exactly zero.
    This is useful for feature selection and preventing overfitting.
    
    Args:
        model (nn.Module): Neural network model
        lambda_l1 (float): Regularization strength (λ)
            - λ = 0: No regularization
            - λ small (0.001-0.01): Light regularization
            - λ large (0.1-1.0): Strong regularization (may underfit)
    
    Returns:
        torch.Tensor: Scalar L1 penalty term
    
    Mathematical Properties:
        - Non-differentiable at 0 (but subgradient exists)
        - Encourages sparse solutions (some weights → 0)
        - Used in Lasso regression
        - Scale: Unbounded (grows linearly with |weights|)
    
    How it works:
        Total Loss = Data Loss + λ * L1
        
        Example: If MSE = 1.0 and L1 = 10.0, then:
        - With λ = 0.01:  Total = 1.0 + 0.01*10.0 = 1.1
        - With λ = 0.1:   Total = 1.0 + 0.1*10.0 = 2.0
        
        Higher λ → stronger penalty → smaller weights → simpler model
    
    Pedagogical Note:
        L1 regularization is like a "sparse prior" in Bayesian terms.
        It assumes weights come from a Laplace distribution:
        
        p(w) ∝ exp(-λ|w|)
        
        Maximum a posteriori (MAP) estimation with this prior
        gives L1 regularization!
    """
    l1_penalty = 0.0
    
    # Sum absolute values of all parameters
    for param in model.parameters():
        l1_penalty += torch.sum(torch.abs(param))
    
    # Scale by regularization strength
    return lambda_l1 * l1_penalty


def l2_regularization(model, lambda_l2=0.01):
    """
    Compute L2 (Ridge) regularization penalty.
    
    L2 = λ * Σwᵢ²
    
    L2 regularization promotes SMALL weights - it shrinks all weights
    toward zero but rarely makes them exactly zero.
    
    Args:
        model (nn.Module): Neural network model
        lambda_l2 (float): Regularization strength (λ)
            - λ = 0: No regularization
            - λ small (0.001-0.01): Light regularization
            - λ large (0.1-1.0): Strong regularization (may underfit)
    
    Returns:
        torch.Tensor: Scalar L2 penalty term
    
    Mathematical Properties:
        - Differentiable everywhere (smooth)
        - Encourages small but non-zero weights
        - Used in Ridge regression
        - Scale: Unbounded (grows quadratically with weights²)
    
    How it works:
        Total Loss = Data Loss + λ * L2
        
        Example: If MSE = 1.0 and L2 = 20.0, then:
        - With λ = 0.01:  Total = 1.0 + 0.01*20.0 = 1.2
        - With λ = 0.1:   Total = 1.0 + 0.1*20.0 = 3.0
        
        The squared term penalizes large weights MORE than small ones:
        - Weight = 0.1 → Penalty = 0.01
        - Weight = 1.0 → Penalty = 1.0
        - Weight = 2.0 → Penalty = 4.0  (2x weight, 4x penalty!)
    
    Pedagogical Note:
        L2 regularization is like a "Gaussian prior" in Bayesian terms.
        It assumes weights come from a Gaussian distribution:
        
        p(w) ∝ exp(-λw²)
        
        Maximum a posteriori (MAP) estimation with this prior
        gives L2 regularization!
        
        This is also called "weight decay" in optimization,
        because it literally decays weights toward zero.
    """
    l2_penalty = 0.0
    
    # Sum squared values of all parameters
    for param in model.parameters():
        l2_penalty += torch.sum(param ** 2)
    
    # Scale by regularization strength
    return lambda_l2 * l2_penalty


def elastic_net_regularization(model, lambda_l1=0.01, lambda_l2=0.01):
    """
    Compute Elastic Net regularization (combination of L1 and L2).
    
    Elastic Net = λ₁ * Σ|wᵢ| + λ₂ * Σwᵢ²
    
    Combines benefits of both L1 and L2:
    - L1 component: Promotes sparsity (some weights → 0)
    - L2 component: Prevents large weights (all weights small)
    
    Args:
        model (nn.Module): Neural network model
        lambda_l1 (float): L1 regularization strength
        lambda_l2 (float): L2 regularization strength
    
    Returns:
        torch.Tensor: Scalar Elastic Net penalty
    
    Why use Elastic Net?
        - Pure L1: Can be unstable with correlated features
        - Pure L2: Doesn't produce sparse solutions
        - Elastic Net: Gets both sparsity AND stability!
    
    Typical settings:
        - Balanced: λ₁ = λ₂ (equal importance)
        - Sparse bias: λ₁ > λ₂ (more L1, more sparsity)
        - Smooth bias: λ₂ > λ₁ (more L2, smoother weights)
    """
    l1_penalty = l1_regularization(model, lambda_l1=lambda_l1)
    l2_penalty = l2_regularization(model, lambda_l2=lambda_l2)
    
    return l1_penalty + l2_penalty


def compute_total_loss_with_regularization(data_loss, model, 
                                          reg_type='none',
                                          lambda_l1=0.0, 
                                          lambda_l2=0.0):
    """
    Compute total loss including data loss and regularization.
    
    This is a convenience function that combines data loss (MSE, CE, etc.)
    with regularization penalties.
    
    Args:
        data_loss (torch.Tensor): Loss on data (MSE, BCE, CE, etc.)
        model (nn.Module): Neural network model
        reg_type (str): Type of regularization
            - 'none': No regularization
            - 'l1': L1 (Lasso) regularization
            - 'l2': L2 (Ridge) regularization  
            - 'elastic': Elastic Net (L1 + L2)
        lambda_l1 (float): L1 regularization strength
        lambda_l2 (float): L2 regularization strength
    
    Returns:
        tuple: (total_loss, reg_penalty)
            - total_loss: data_loss + reg_penalty
            - reg_penalty: regularization penalty (for logging)
    
    Example:
        >>> data_loss = mean_squared_error(y_pred, y_true)
        >>> total_loss, reg_penalty = compute_total_loss_with_regularization(
        ...     data_loss, model, reg_type='l2', lambda_l2=0.01
        ... )
        >>> # Backpropagate on total_loss, not just data_loss!
        >>> total_loss.backward()
    
    Pedagogical Note:
        This function makes it explicit that:
        
        Total Loss = Data Fit Term + Complexity Penalty
        
        The data fit term (MSE, etc.) encourages good predictions.
        The complexity penalty (L1/L2) encourages simple models.
        
        This is the bias-variance tradeoff in action:
        - No regularization (λ=0): Low bias, high variance (overfit)
        - Strong regularization (λ large): High bias, low variance (underfit)
        - Optimal λ: Balances bias and variance
    """
    if reg_type == 'none' or (lambda_l1 == 0.0 and lambda_l2 == 0.0):
        return data_loss, torch.tensor(0.0)
    
    elif reg_type == 'l1':
        reg_penalty = l1_regularization(model, lambda_l1=lambda_l1)
        total_loss = data_loss + reg_penalty
        return total_loss, reg_penalty
    
    elif reg_type == 'l2':
        reg_penalty = l2_regularization(model, lambda_l2=lambda_l2)
        total_loss = data_loss + reg_penalty
        return total_loss, reg_penalty
    
    elif reg_type == 'elastic':
        reg_penalty = elastic_net_regularization(model, lambda_l1=lambda_l1, lambda_l2=lambda_l2)
        total_loss = data_loss + reg_penalty
        return total_loss, reg_penalty
    
    else:
        raise ValueError(f"Unknown reg_type: {reg_type}. Choose from: 'none', 'l1', 'l2', 'elastic'")


def demonstrate_regularization_effect():
    """
    Educational demonstration of how regularization affects weights.
    """
    logger.info("\n" + "="*70)
    logger.info("DEMONSTRATION: Regularization Effects on Weights")
    logger.info("="*70)
    
    # Create a simple model with large weights
    from MultiLayerPerceptron import MultiLayerPerceptron
    model = MultiLayerPerceptron([1, 10, 1], activation='tanh')
    
    # Manually set some large weights to demonstrate
    with torch.no_grad():
        model.layers[0].weight.fill_(2.0)  # Large weights
        model.layers[1].weight.fill_(2.0)
    
    logger.info("\nOriginal weights (intentionally large):")
    logger.info(f"  Layer 1 weights: ~2.0")
    logger.info(f"  Layer 2 weights: ~2.0")
    
    # Compute penalties
    l1_pen = l1_regularization(model, lambda_l1=0.1)
    l2_pen = l2_regularization(model, lambda_l2=0.1)
    
    logger.info(f"\nRegularization penalties:")
    logger.info(f"  L1 penalty (λ=0.1): {l1_pen.item():.4f}")
    logger.info(f"  L2 penalty (λ=0.1): {l2_pen.item():.4f}")
    
    logger.info("\nEffect during training:")
    logger.info("  L1: Gradient ∝ sign(w) → pushes weights toward 0, creates sparsity")
    logger.info("  L2: Gradient ∝ w → shrinks large weights more than small ones")
    
    # Show weight distribution
    all_weights = torch.cat([p.flatten() for p in model.parameters()])
    logger.info(f"\nWeight statistics:")
    logger.info(f"  Mean: {all_weights.mean():.4f}")
    logger.info(f"  Std:  {all_weights.std():.4f}")
    logger.info(f"  Max:  {all_weights.max():.4f}")
    logger.info(f"  Min:  {all_weights.min():.4f}")
    
    logger.info("\n" + "="*70)
    logger.info("KEY INSIGHT: Regularization penalizes model complexity!")
    logger.info("This prevents overfitting by preferring simpler models.")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    # Demo when run directly
    from logger import configure_logging
    configure_logging()
    
    compare_losses_mathematically()
    demonstrate_loss_behavior()
    demonstrate_regularization_effect()


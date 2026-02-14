"""
Tutorial 2: Perceptron to DNN

This package implements a simple perceptron, and then expands this into
a layered set of perceptron, a multi-layer perceptron (MLP), or Dense
Neural Network (DNN). This tutorial is for educational purposes
"""

__version__ = "0.1.0"

from .train import train_model_with_gradient_tracking, train_model_with_validation_tracking
from .utils import FeatureNormalizer
from .plotting import (plot_gradient_flow, plot_layer_gradient_norms, 
                       plot_regularization_comparison, plot_results)

__all__ = [
    "train_model_with_gradient_tracking",
    "train_model_with_validation_tracking",
    "FeatureNormalizer",
    "plot_gradient_flow",
    "plot_layer_gradient_norms",
    "plot_regularization_comparison",
    "plot_results"
    ]

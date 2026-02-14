"""
Source package for L38 CNN Odd Number Detector.

This package contains all modules for the CNN image classification pipeline:
- data_loader: Load images, assign labels, create DataLoaders
- model: CNN architecture definition
- trainer: Training loop with validation
- evaluator: Test evaluation and metrics
- visualizer: Training and prediction plots
- eval_plots: Evaluation plots (confusion matrix, sample images)
"""

from .data_loader import load_dataset
from .model import OddNumberCNN, TransferCNN
from .trainer import train_model
from .evaluator import evaluate_model
from .visualizer import (
    plot_loss_curve,
    plot_accuracy_curve,
    plot_prediction_grid,
)
from .eval_plots import plot_confusion_matrix, plot_sample_images

__all__ = [
    "load_dataset",
    "OddNumberCNN",
    "TransferCNN",
    "train_model",
    "evaluate_model",
    "plot_loss_curve",
    "plot_accuracy_curve",
    "plot_prediction_grid",
    "plot_confusion_matrix",
    "plot_sample_images",
]

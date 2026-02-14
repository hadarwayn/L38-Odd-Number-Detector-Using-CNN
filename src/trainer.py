"""
Training Module for CNN Odd Number Detector.

Training loop: Forward Pass -> Loss (BCEWithLogitsLoss) -> Backprop -> Adam.
Uses ReduceLROnPlateau scheduler and best-model checkpoint tracking.

BCEWithLogitsLoss combines sigmoid + BCE in one numerically stable step.
Adam adapts per-parameter learning rates using momentum + RMSprop benefits.
"""

import logging
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Classification threshold: probability >= 0.5 → predicted class 1
CLASSIFICATION_THRESHOLD = 0.5


def _compute_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Calculate classification accuracy on a dataset.

    Runs the model in evaluation mode (no gradient tracking) and
    compares predictions against true labels.
    """
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            predictions = (torch.sigmoid(outputs) >= CLASSIFICATION_THRESHOLD)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 15,
    learning_rate: float = 0.001,
    device: torch.device | None = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the CNN model and track metrics per epoch.

    Args:
        model: The OddNumberCNN instance.
        train_loader: DataLoader for training batches.
        test_loader: DataLoader for validation/test batches.
        num_epochs: How many full passes through the training data.
        learning_rate: Step size for the Adam optimizer.
        device: CPU or CUDA device.

    Returns:
        Tuple of (trained_model, history_dict) where history contains
        'train_loss' and 'test_accuracy' lists, one value per epoch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "test_accuracy": [],
    }

    best_accuracy = 0.0
    best_state = None

    start_time = time.time()
    logger.info("Training on %s for %d epochs (lr=%.4f)",
                device, num_epochs, learning_rate)

    for epoch in range(1, num_epochs + 1):
        # --- Training phase ---
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # --- Validation phase ---
        accuracy = _compute_accuracy(model, test_loader, device)

        history["train_loss"].append(avg_loss)
        history["test_accuracy"].append(accuracy)
        scheduler.step(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        logger.info("Epoch %2d/%d — Loss: %.4f — Accuracy: %.2f%%",
                     epoch, num_epochs, avg_loss, accuracy * 100)

    # Restore best model weights
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("Restored best model (%.2f%% accuracy)", best_accuracy * 100)

    total_time = time.time() - start_time
    logger.info("Training complete in %.1f seconds", total_time)
    history["training_time_seconds"] = total_time

    return model, history

"""
CNN Model Architecture for Odd Number Detector.

Two model options:
1. OddNumberCNN - custom 3-layer CNN built from scratch (Lesson 38)
2. TransferCNN  - pre-trained ResNet18 backbone fine-tuned for our task

The transfer learning model adapts a network pre-trained on ImageNet
(millions of images) to our small dataset. It already knows edges,
shapes, and textures — we just teach it our specific classification.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = logging.getLogger(__name__)

# Custom CNN constants
CONV1_OUT = 32
CONV2_OUT = 64
CONV3_OUT = 128
KERNEL_SIZE = 3
POOL_SIZE = 2
FC_HIDDEN = 128
FLAT_FEATURES = CONV3_OUT * 8 * 8  # = 8,192


class OddNumberCNN(nn.Module):
    """Custom 3-layer CNN for binary classification."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, CONV1_OUT, KERNEL_SIZE, padding=1)
        self.bn1 = nn.BatchNorm2d(CONV1_OUT)
        self.conv2 = nn.Conv2d(CONV1_OUT, CONV2_OUT, KERNEL_SIZE, padding=1)
        self.bn2 = nn.BatchNorm2d(CONV2_OUT)
        self.conv3 = nn.Conv2d(CONV2_OUT, CONV3_OUT, KERNEL_SIZE, padding=1)
        self.bn3 = nn.BatchNorm2d(CONV3_OUT)
        self.pool = nn.MaxPool2d(POOL_SIZE, POOL_SIZE)
        self.fc1 = nn.Linear(FLAT_FEATURES, FC_HIDDEN)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(FC_HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: image pixels -> prediction logit."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_summary(self) -> None:
        """Print a human-readable model summary."""
        logger.info("Model: OddNumberCNN (3-layer + BatchNorm)")
        logger.info("  Total: %s parameters", f"{self.count_parameters():,}")


class TransferCNN(nn.Module):
    """ResNet18 backbone fine-tuned for odd/even classification."""

    def __init__(self) -> None:
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Adapt first layer: 3-channel RGB -> 1-channel grayscale
        orig = backbone.conv1
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.copy_(orig.weight.mean(dim=1, keepdim=True))

        # Reuse all ResNet layers except first conv and final FC
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet18 backbone -> binary logit."""
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_summary(self) -> None:
        """Print a human-readable model summary."""
        logger.info("Model: TransferCNN (ResNet18 backbone)")
        logger.info("  Total: %s parameters", f"{self.count_parameters():,}")

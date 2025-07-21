"""
Callbacks module for PyTorch Lightning.

This module provides custom callbacks for training and evaluation of deep learning models.
"""

from .map import MeanAveragePrecisionCallback
from .wandb import WandbObjectDetectionCallback

__all__ = [
    "MeanAveragePrecisionCallback",
    "WandbObjectDetectionCallback",
]

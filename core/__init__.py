"""
Core modules for the XBD damage classification system.

This package contains the fundamental components:
- config: configuration classes for training, inference and XAI
- dataset: dataset loader and data augmentation
- models: neural network architectures
- losses: custom loss functions
- utils: common utility functions
"""

from .config import TrainingConfig, InferenceConfig, SegmentationConfig
from .dataset import (BuildingDamageDatasetHDF5, data_transforms,
                      create_training_dataset, create_validation_dataset,
                      SiameseSyncTransform, NoAugmentTransform)
from .models import SiameseNetwork, load_model_from_checkpoint
from .losses import FocalLoss, create_loss_function
from .utils import Logger, setup_directories

__all__ = [
    'TrainingConfig',
    'InferenceConfig',
    'SegmentationConfig',
    'BuildingDamageDatasetHDF5',
    'data_transforms',
    'create_training_dataset',
    'create_validation_dataset',
    'SiameseSyncTransform',
    'NoAugmentTransform',
    'SiameseNetwork',
    'load_model_from_checkpoint',
    'FocalLoss',
    'create_loss_function',
    'Logger',
    'setup_directories',
]

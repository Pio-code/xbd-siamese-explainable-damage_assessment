"""
Data loading and transformations for xBD dataset.

Contains:
- SiameseSyncTransform: Synchronized data augmentation for siamese pairs
- NoAugmentTransform: Transformations without augmentation
- BuildingDamageDatasetHDF5: PyTorch dataset for loading from HDF5
- create_training_dataset
- create_validation_dataset
"""

import os
import h5py
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF


class SiameseSyncTransform:
    """
    Class to apply identical transformations to image pairs.
   
    In siamese architectures, the same random transformation must be applied
    to both pre and post images to maintain geometric and color consistency.
    Parameters defined in TrainingConfig.
    """
    
    def __init__(self, 
                 horizontal_flip_prob,
                 vertical_flip_prob, 
                 rotation_prob,
                 color_jitter_prob,
                 brightness,
                 contrast,
                 saturation,
                 hue):

        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.rotation_prob = rotation_prob
        self.color_jitter_prob = color_jitter_prob
        
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        
        # Base transformation (always applied)
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __call__(self, image_pre_np, image_post_np):
        """
        Applies the same random transformations to both images.
        """
        # Convert from numpy to PIL for transformations
        image_pre_pil = Image.fromarray(image_pre_np.astype(np.uint8))
        image_post_pil = Image.fromarray(image_post_np.astype(np.uint8))
        
        if random.random() < self.horizontal_flip_prob:
            image_pre_pil = TF.hflip(image_pre_pil)
            image_post_pil = TF.hflip(image_post_pil)
            
        if random.random() < self.vertical_flip_prob:
            image_pre_pil = TF.vflip(image_pre_pil)
            image_post_pil = TF.vflip(image_post_pil)
            
        if random.random() < self.rotation_prob:
            rotation_angle = random.choice([90, 180, 270])
            image_pre_pil = TF.rotate(image_pre_pil, rotation_angle)
            image_post_pil = TF.rotate(image_post_pil, rotation_angle)
            
        if random.random() < self.color_jitter_prob:
            random_state = torch.get_rng_state()
            
            torch.set_rng_state(random_state)
            image_pre_pil = self.color_jitter(image_pre_pil)
            
            torch.set_rng_state(random_state) 
            image_post_pil = self.color_jitter(image_post_pil)
        
        image_pre_tensor = self.base_transform(image_pre_pil)
        image_post_tensor = self.base_transform(image_post_pil)
        
        return image_pre_tensor, image_post_tensor


class NoAugmentTransform:
    """
    Transformation without augmentation for validation/test.
    Applies only ToTensor and ImageNet Normalization.
    """
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __call__(self, image_pre_np, image_post_np):
        image_pre_pil = Image.fromarray(image_pre_np.astype(np.uint8))
        image_post_pil = Image.fromarray(image_post_np.astype(np.uint8))
        
        image_pre_tensor = self.transform(image_pre_pil)
        image_post_tensor = self.transform(image_post_pil)
        
        return image_pre_tensor, image_post_tensor


class BuildingDamageDatasetHDF5(Dataset):
    """
    Dataset class to load patch pairs from a single HDF5 file.
    
    Use helper functions to create configured instances:
    - create_training_dataset(path, enable_augmentation=True, config=config)
    - create_validation_dataset(path)
    """
    def __init__(self, hdf5_path, transform=None):
        """
        Args:
            hdf5_path (str): Path to the pre-processed HDF5 file.
            transform: Transformations to apply (optional, default: NoAugmentTransform)
        """
        self.hdf5_path = hdf5_path
        
        if transform is not None:
            print(f" Custom transformations for {hdf5_path}")
            self.transform = transform
        else:
            print(f" Base transformations (NoAugment) for {hdf5_path}")
            self.transform = NoAugmentTransform()
        
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f" HDF5 FILE NOT FOUND: {hdf5_path}")
        print(f" HDF5 file found: {hdf5_path}")
        
        # Don't open file in __init__ for multiprocessing compatibility.
        # Cannot be easily transferred between processes.
        self.hdf5_file = None
        self.pre_images = None
        self.post_images = None
        self.labels = None
        
        # Get only the length by opening and closing the file
        # For the Dataset class to work, PyTorch needs to know immediately
        # its total length (via __len__). So the file is opened
        # briefly just to read this information and then closed.
        with h5py.File(self.hdf5_path, 'r') as f:
            self.length = len(f['labels'])
            print(f" HDF5 dataset contains {self.length:,} samples")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r')
            self.pre_images = self.hdf5_file['pre_images']
            self.post_images = self.hdf5_file['post_images']
            self.labels = self.hdf5_file['labels']
        
        try:
            image_pre = self.pre_images[idx]
            image_post = self.post_images[idx]
            label = self.labels[idx]

            image_pre, image_post = self.transform(image_pre, image_post)
                
            return image_pre, image_post, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f" Error loading sample {idx}: {e}")
            return None 

    def close(self):
        """
        Method to close the HDF5 file when finished.
        """
        if self.hdf5_file:
            self.hdf5_file.close()

# TRANSFORMATION FOR SINGLE IMAGES
# Used by XAI scripts (LIME, SHAP, GradCAM) to process
# single images or masks generated during explainability.
# Not to be confused with SiameseSyncTransform which processes image pairs.
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# UTILITY FUNCTIONS FOR DATA AUGMENTATION
def create_training_dataset(hdf5_path, enable_augmentation=True, config=None):
    """
    Creates a dataset for training with optional data augmentation.

    Args:
        hdf5_path (str): Path to HDF5 file
        enable_augmentation (bool): Whether to enable data augmentation
        config (TrainingConfig): Configuration with augmentation parameters
    
    Returns:
        BuildingDamageDatasetHDF5: Configured dataset
    """
    if enable_augmentation:
        if config is None:
            raise ValueError("config is required when enable_augmentation=True")
        
        # Create transformation with parameters from config
        transform = SiameseSyncTransform(
            horizontal_flip_prob=config.augment_horizontal_flip_prob,
            vertical_flip_prob=config.augment_vertical_flip_prob,
            rotation_prob=config.augment_rotation_prob,
            color_jitter_prob=config.augment_color_jitter_prob,
            brightness=config.augment_brightness,
            contrast=config.augment_contrast,
            saturation=config.augment_saturation,
            hue=config.augment_hue
        )
        return BuildingDamageDatasetHDF5(hdf5_path, transform=transform)
    else:
        # Use NoAugmentTransform (default)
        return BuildingDamageDatasetHDF5(hdf5_path)


def create_validation_dataset(hdf5_path):
    """
    Creates a dataset for validation without data augmentation.
    
    Args:
        hdf5_path (str): Path to HDF5 file
    
    Returns:
        BuildingDamageDatasetHDF5: Dataset with NoAugmentTransform
    """
    return BuildingDamageDatasetHDF5(hdf5_path)

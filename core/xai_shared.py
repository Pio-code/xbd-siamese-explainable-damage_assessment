"""
Shared functions for XAI analysis (LIME, Grad-CAM, etc.).

This module contains all duplicated functions across XAI scripts:
- Image denormalization
- Creating prediction functions for LIME
- Superpixel masking
- Heatmap generation
- Creating segmentation functions
"""

import torch
import numpy as np
from typing import Callable, List, Optional
from PIL import Image


def denormalize_image(tensor: torch.Tensor, 
                     mean: tuple = (0.485, 0.456, 0.406),
                     std: tuple = (0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Denormalizes an image tensor for XAI processing.
    
    XAI VERSION: For masks, perturbations, and heatmap generation.
    - Input: single image [C, H, W] (NO batch)
    - Output: float [0, 1] for further processing
    
    Args:
        tensor: PyTorch tensor [C, H, W] normalized (NO batch dimension)
        mean: Mean used for normalization (default: ImageNet)
        std: Standard deviation used for normalization (default: ImageNet)
    
    Returns:
        np.ndarray: Float image [0, 1], shape (H, W, C)
    """
    # Convert to numpy and transpose from CHW to HWC
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    
    # Denormalize
    mean = np.array(mean)
    std = np.array(std)
    img = std * img + mean
    
    # Clip for safety (might go outside [0,1] due to rounding)
    img = np.clip(img, 0, 1)
    
    return img


def create_prediction_function_for_lime(model: torch.nn.Module,
                                       pre_image_tensor: torch.Tensor,
                                       device: torch.device,
                                       data_transforms) -> Callable:
    """
    Creates prediction function for LIME.
    
    The PRE image remains FIXED (anchor), while LIME perturbs only POST.
    
    Args:
        model: Siamese model
        pre_image_tensor: PRE tensor [1, C, H, W] (fixed ANCHOR)
        device: Device (cuda/cpu)
        data_transforms: Transforms to apply to perturbations
    
    Returns:
        Callable: Function predict_fn(post_images_np) -> probabilities_np
    """
    def predict_fn(post_images_perturbed_np: np.ndarray) -> np.ndarray:
        """
        Predicts for batch of perturbed POST images.
        
        Args:
            post_images_perturbed_np: [N, H, W, C] numpy array [0, 1]
        
        Returns:
            np.ndarray: [N, num_classes] probabilities
        """
        num_perturbations = post_images_perturbed_np.shape[0]
        
        # Expand PRE to match batch size
        batch_pre_tensor = pre_image_tensor.expand(num_perturbations, -1, -1, -1)
        
        # Convert perturbed POST: numpy -> PIL -> tensor
        batch_post_tensor_list = [
            data_transforms(Image.fromarray((img * 255).astype(np.uint8))) 
            for img in post_images_perturbed_np
        ]
        batch_post_tensor = torch.stack(batch_post_tensor_list).to(device)
        
        # Prediction
        with torch.no_grad():
            outputs = model(batch_pre_tensor, batch_post_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()
    
    return predict_fn


def create_anchor_and_perturb_prediction_function(
        model: torch.nn.Module,
        pre_image_tensor: torch.Tensor,
        device: torch.device,
        data_transforms) -> Callable:
    """
    Alias for create_prediction_function_for_lime (for backward compatibility).
    
    Used in explain_* scripts that use the more descriptive name.
    """
    return create_prediction_function_for_lime(
        model, pre_image_tensor, device, data_transforms
    )


def create_prediction_function_for_shap(
        model: torch.nn.Module,   
        pre_image_tensor: torch.Tensor, 
        post_image_viz: np.ndarray, # Original POST image
        device: torch.device,
        data_transforms,
        segments: np.ndarray, #  Superpixel map
        background_mean: np.ndarray) -> Callable:  # Baseline for "absent" superpixels
    """
    Creates prediction function for KernelSHAP.
    
    The pre image remains fixed, while SHAP perturbs POST using
    superpixel coalitions. "Absent" superpixels are replaced with
    background mean (neutral baseline).
    
    Strategy: "Anchor and Perturb" (same as LIME)
    - PRE: Fixed during all perturbations
    - POST: "Present" superpixels (mask=1) use real pixels
            "Absent" superpixels (mask=0) use background_mean
    
    Args:
        model: Siamese model
        pre_image_tensor: PRE tensor [1, C, H, W] (fixed ANCHOR)
        post_image_viz: POST image denormalized [H, W, C] in [0, 1]
        device: Device (cuda/cpu)
        data_transforms: Transforms to apply to perturbations
        segments: Segmentation map [H, W] with superpixel IDs
        background_mean: Background mean [H, W, C] in [0, 1] (baseline)
    
    Returns:
        Callable: Function predict_fn(binary_mask) -> probabilities_np
    """
    def predict_fn(binary_mask: np.ndarray) -> np.ndarray:
        """
        Predicts for batch of superpixel coalitions.
        
        Args:
            binary_mask: [N, num_superpixels] array with 0/1
        
        Returns:
            np.ndarray: [N, num_classes] probabilities
        """
        batch_size = binary_mask.shape[0]
        batch_pre = pre_image_tensor.expand(batch_size, -1, -1, -1)
        batch_post_list = []
        
        # For each superpixel coalition
        for mask_row in binary_mask:
            # Start with baseline (all superpixels absent)
            masked_image = background_mean.copy()
            
            # Add only superpixels in coalition (mask=1)
            for sp_id in range(len(mask_row)):
                if mask_row[sp_id] == 1:  # Superpixel present
                    sp_mask = (segments == sp_id)
                    masked_image[sp_mask] = post_image_viz[sp_mask]  # Real pixels
            
            # Convert to PyTorch tensor
            masked_pil = Image.fromarray((masked_image * 255).astype(np.uint8))
            masked_tensor = data_transforms(masked_pil)
            batch_post_list.append(masked_tensor)
        
        batch_post = torch.stack(batch_post_list).to(device)
        
        # Model prediction
        with torch.no_grad():
            outputs = model(batch_pre, batch_post)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()
    
    return predict_fn



def mask_superpixels(image: np.ndarray, 
                    segments: np.ndarray, 
                    superpixel_ids: List[int], 
                    mask_value: float = 0.0) -> np.ndarray:
    """
    Masks specified superpixels by setting their pixels to mask_value.
    
    Args:
        image: Image [H, W, C] in [0, 1]
        segments: Segmentation map [H, W] with superpixel IDs
        superpixel_ids: List of superpixel IDs to mask
        mask_value: Value to set (default: 0 = black)
    
    Returns:
        np.ndarray: Masked image [H, W, C]
    
    Note:
        - Used for deletion/insertion curves
        - Copies image to avoid side effects
        - mask_value=0 for deletion, =mean for insertion
    """
    masked_image = image.copy()
    
    for sp_id in superpixel_ids:
        mask = (segments == sp_id)
        masked_image[mask] = mask_value
    
    return masked_image



def create_lime_superpixel_heatmap(base_image: np.ndarray, 
                                  segments: np.ndarray, 
                                  feature_importance: List[tuple],
                                  alpha: float = 0.5) -> np.ndarray:
    """
    Creates colored heatmap based on LIME superpixel importances.
    
    Color scheme:
        - Green: POSITIVE contribution (supporting target class)
        - Red: NEGATIVE contribution (against target class)
        - Color intensity proportional to importance
    
    Args:
        base_image: Base image [H, W, C] in [0, 1]
        segments: Segmentation map [H, W]
        feature_importance: List of (superpixel_id, importance_weight)
        alpha: Overlay transparency (0=only image, 1=only heatmap)
    
    Returns:
        np.ndarray: Image with heatmap overlay [H, W, C] in [0, 1]
    """
    importance_map = np.zeros(segments.shape, dtype=np.float32)
    
    for superpixel_id, weight in feature_importance:
        mask = (segments == superpixel_id)
        importance_map[mask] = weight
    
    abs_max = np.abs(importance_map).max()
    if abs_max > 0:
        importance_map = importance_map / abs_max  # Range [-1, 1]
    
    heatmap_colored = np.zeros((segments.shape[0], segments.shape[1], 3), dtype=np.float32)
    
    positive_mask = importance_map > 0
    heatmap_colored[positive_mask, 1] = importance_map[positive_mask]  # Green channel
    
    negative_mask = importance_map < 0
    heatmap_colored[negative_mask, 0] = -importance_map[negative_mask]  # Red channel (negative becomes positive)
    
    base_image_float = base_image.astype(np.float32)
    
    blended = alpha * heatmap_colored + (1 - alpha) * base_image_float
    
    blended = np.clip(blended, 0, 1)
    
    return blended


def create_shap_superpixel_heatmap(base_image: np.ndarray,
                                   segments: np.ndarray,
                                   shap_values: np.ndarray,
                                   alpha: float = 0.5) -> np.ndarray:
    """
    Creates colored heatmap based on Shapley values of superpixels.
    
    Color scheme:
        - Green: POSITIVE contribution (supporting target class)
        - Red: NEGATIVE contribution (against target class)
        - Color intensity proportional to importance
    
    Args:
        base_image: Base image [H, W, C] in [0, 1]
        segments: Segmentation map [H, W]
        shap_values: Array [num_superpixels] with Shapley values
        alpha: Overlay transparency (0=only image, 1=only heatmap)
    
    Returns:
        np.ndarray: Image with heatmap overlay [H, W, C] in [0, 1]
    
    Note:
        - Logic is identical to create_lime_superpixel_heatmap
        - Difference: SHAP uses direct array, LIME uses list of tuples
        - Same normalization and color scheme for visual comparison
    """
    importance_map = np.zeros(segments.shape, dtype=np.float32)
    
    for sp_id in range(len(shap_values)):
        mask = (segments == sp_id)
        importance_map[mask] = shap_values[sp_id]
    
    abs_max = np.abs(importance_map).max()
    if abs_max > 0:
        importance_map = importance_map / abs_max  # Range [-1, 1]
    
    heatmap_colored = np.zeros((segments.shape[0], segments.shape[1], 3), dtype=np.float32)
    
    # Positive values -> green (G channel)
    positive_mask = importance_map > 0
    heatmap_colored[positive_mask, 1] = importance_map[positive_mask]
    
    # Negative values -> red (R channel)
    negative_mask = importance_map < 0
    heatmap_colored[negative_mask, 0] = -importance_map[negative_mask]
    

    base_image_float = base_image.astype(np.float32)
    
    blended = alpha * heatmap_colored + (1 - alpha) * base_image_float
    
    blended = np.clip(blended, 0, 1)
    
    return blended


# GRAD-CAM WRAPPERS (deprecated)

class SiameseWrapperForGradCAM(torch.nn.Module):
    """
    Wrapper for Siamese Network compatible with pytorch-grad-cam.
    
    Grad-CAM expects forward(x) with single input, but Siamese has
    forward(pre, post) with two inputs. This wrapper concatenates the two
    inputs on channels and then splits them internally.
    
    Note:
        - Input: [B, 6, H, W] (3 channels PRE + 3 channels POST)
        - Split: [B, 3, H, W] for PRE and POST
        - Output: [B, num_classes] logits
    """
    def __init__(self, siamese_model: torch.nn.Module):
        super().__init__()
        self.model = siamese_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with concatenated input.
        
        Args:
            x: Tensor [B, 6, H, W] (PRE and POST concatenated)
        
        Returns:
            Tensor [B, num_classes] logits
        """
        # Split on channels: first 3 = PRE, last 3 = POST
        pre = x[:, :3, :, :]
        post = x[:, 3:, :, :]
        
        # Normal forward
        return self.model(pre, post)



def get_top_superpixels_by_importance(feature_importance: List[tuple],
                                     top_n: int = 10,
                                     sort_by_abs: bool = True) -> List[tuple]:
    """
    Get top-N superpixels by importance.
    
    Args:
        feature_importance: List of (superpixel_id, weight)
        top_n: Number of top superpixels to return
        sort_by_abs: If True, sort by |weight| (absolute importance)
                     If False, sort by weight (keeps sign)
    
    Returns:
        List[tuple]: Top-N superpixels sorted
    """
    if sort_by_abs:
        sorted_features = sorted(
            feature_importance, 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
    else:
        # Sort by weight (positives first, then negatives)
        sorted_features = sorted(
            feature_importance, 
            key=lambda x: x[1], 
            reverse=True
        )
    
    return sorted_features[:top_n]


# MODULE INFO

__all__ = [
    # Image preprocessing
    'denormalize_image',
    
    # LIME - Prediction
    'create_prediction_function_for_lime',
    'create_anchor_and_perturb_prediction_function',
    
    # SHAP - Prediction
    'create_prediction_function_for_shap',
    
    # Superpixel manipulation
    'mask_superpixels',
    
    # Visualization (LIME & SHAP)
    'create_lime_superpixel_heatmap',
    'create_shap_superpixel_heatmap',
    
    # Grad-CAM wrappers
    'SiameseWrapperForGradCAM',
    
    # Utilities
    'get_top_superpixels_by_importance'
]

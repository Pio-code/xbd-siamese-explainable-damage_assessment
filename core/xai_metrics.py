"""
Evaluation metrics for XAI methods.

This module contains:
- Deletion and Insertion curves with AUC calculation for LIME/SHAP
- Specific logic: removes/adds only POSITIVE superpixels (supporting the prediction)
"""

import numpy as np
import torch
from typing import List, Dict
from sklearn.metrics import auc as sklearn_auc


def compute_lime_deletion_insertion_curves(
        model: torch.nn.Module,
        pre_tensor: torch.Tensor,
        post_tensor: torch.Tensor,
        post_image_viz: np.ndarray,
        segments: np.ndarray,
        feature_importance: List[tuple],
        target_class: int,
        device: torch.device,
        idx_to_class: Dict[int, str],
        mask_value: float = 0.0):
    """
    Calculates deletion and insertion curves for LIME/SHAP using only positive superpixels.
    
    Args:
        model: Siamese model to evaluate
        pre_tensor: PRE image [1, C, H, W] normalized
        post_tensor: POST image [1, C, H, W] normalized
        post_image_viz: POST image denormalized [H, W, C] in [0, 1]
        segments: Segmentation map [H, W]
        feature_importance: List (superpixel_id, weight) from LIME
        target_class: Target class index (predicted_idx)
        device: Device (cuda/cpu)
        idx_to_class: Dictionary {idx: class_name}
        mask_value: Value for masked pixels (default: 0 = black)
    
    Returns:
        Dict with:
            - 'deletion_curves': Dict[class_name, {'percentages': array, 'confidences': list}]
            - 'insertion_curves': Dict[class_name, {'percentages': array, 'confidences': list}]
            - 'auc_scores': Dict[class_name, {'deletion_auc': float, 'insertion_auc': float}]
            - 'num_positive_superpixels': int
    """
    from core.xai_shared import mask_superpixels
    from core.dataset import data_transforms
    from PIL import Image
    
    with torch.no_grad():
        output = model(pre_tensor, post_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
    
    sorted_superpixels = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)
    positive_superpixels = [sp_id for sp_id, weight in sorted_superpixels if weight > 0]
    
    num_positive = len(positive_superpixels)
    if num_positive == 0:
        raise ValueError("No positive superpixels found. Cannot calculate curves.")
    
    actual_steps = num_positive + 1  # +1 to include 0% step
    percentages_of_positive = np.linspace(0, 1.0, actual_steps)
    
    results = {
        'deletion_curves': {},
        'insertion_curves': {},
        'auc_scores': {},
        'num_positive_superpixels': num_positive
    }
    
    for class_idx, class_name in idx_to_class.items():
        results['deletion_curves'][class_name] = {
            'percentages': percentages_of_positive,
            'confidences': [],
            'original_confidence': probabilities[0, class_idx].item()
        }
        results['insertion_curves'][class_name] = {
            'percentages': percentages_of_positive,
            'confidences': [],
            'original_confidence': probabilities[0, class_idx].item()
        }
    
    # CALCULATE DELETION CURVES 
    for pct in percentages_of_positive:
        num_to_remove = int(pct * num_positive)
        
        if num_to_remove == 0:
            # No removal = original confidences
            for class_idx, class_name in idx_to_class.items():
                results['deletion_curves'][class_name]['confidences'].append(
                    probabilities[0, class_idx].item()
                )
        else:
            # Remove top positive superpixels
            superpixels_to_remove = positive_superpixels[:num_to_remove]
            masked_post_image = mask_superpixels(post_image_viz, segments, superpixels_to_remove, mask_value)
            
            # Convert to tensor
            masked_post_tensor = data_transforms(
                Image.fromarray((masked_post_image * 255).astype(np.uint8))
            ).unsqueeze(0).to(device)
            
            # Prediction
            with torch.no_grad():
                masked_output = model(pre_tensor, masked_post_tensor)
                masked_probs = torch.nn.functional.softmax(masked_output, dim=1)
                
                for class_idx, class_name in idx_to_class.items():
                    confidence = masked_probs[0, class_idx].item()
                    results['deletion_curves'][class_name]['confidences'].append(confidence)
    
    # CALCULATE INSERTION CURVES
    for pct in percentages_of_positive:
        num_to_keep = int(pct * num_positive)
        
        if num_to_keep == 0:
            completely_masked_image = np.zeros_like(post_image_viz)
            masked_post_tensor = data_transforms(
                Image.fromarray((completely_masked_image * 255).astype(np.uint8))
            ).unsqueeze(0).to(device)
            
            with torch.no_grad():
                masked_output = model(pre_tensor, masked_post_tensor)
                masked_probs = torch.nn.functional.softmax(masked_output, dim=1)
                
                for class_idx, class_name in idx_to_class.items():
                    confidence = masked_probs[0, class_idx].item()
                    results['insertion_curves'][class_name]['confidences'].append(confidence)
        
        elif pct >= 1.0:
            # Full image = original confidences
            for class_idx, class_name in idx_to_class.items():
                results['insertion_curves'][class_name]['confidences'].append(
                    probabilities[0, class_idx].item()
                )
        else:
            # Keep only top positive superpixels, remove all others
            superpixels_to_keep = positive_superpixels[:num_to_keep]
            all_superpixel_ids = [sp_id for sp_id, _ in sorted_superpixels]
            superpixels_to_remove = [sp_id for sp_id in all_superpixel_ids if sp_id not in superpixels_to_keep]
            
            masked_post_image = mask_superpixels(post_image_viz, segments, superpixels_to_remove, mask_value)
            
            masked_post_tensor = data_transforms(
                Image.fromarray((masked_post_image * 255).astype(np.uint8))
            ).unsqueeze(0).to(device)
            
            with torch.no_grad():
                masked_output = model(pre_tensor, masked_post_tensor)
                masked_probs = torch.nn.functional.softmax(masked_output, dim=1)
                
                for class_idx, class_name in idx_to_class.items():
                    confidence = masked_probs[0, class_idx].item()
                    results['insertion_curves'][class_name]['confidences'].append(confidence)
    
    # CALCULATE AUC FOR ALL CLASSES
    for class_name in idx_to_class.values():
        deletion_confidences = results['deletion_curves'][class_name]['confidences']
        deletion_auc = sklearn_auc(percentages_of_positive, deletion_confidences)
        
        insertion_confidences = results['insertion_curves'][class_name]['confidences']
        insertion_auc = sklearn_auc(percentages_of_positive, insertion_confidences)
        
        results['auc_scores'][class_name] = {
            'deletion_auc': deletion_auc,
            'insertion_auc': insertion_auc
        }
    
    return results



__all__ = [
    'compute_lime_deletion_insertion_curves',
]

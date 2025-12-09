"""
Custom loss functions for the XBD damage classification system.

Contains:
- FocalLoss: Focal Loss for imbalanced classes
- compute_class_weights: Automatic class weight calculation
- create_loss_function: Factory to create loss functions
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


class FocalLoss(nn.Module):
    """
    Focal Loss to handle class imbalance.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (torch.Tensor or None): Class weights. None = no alpha weighting
            gamma (float): Focusing parameter. Typical values: 2.0
            reduction (str): Reduction type ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits from model, shape (batch_size, num_classes)
            targets (torch.Tensor): Target labels, shape (batch_size,)
        
        Returns:
            torch.Tensor: Calculated loss
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        
        # pt is the probability of the correct class
        pt = torch.exp(-ce_loss)
        
        # This is the focusing modulation term
        # Easy examples (pt close to 1) have low weight
        # Hard examples (pt close to 0) have high weight
        focal_term = (1 - pt) ** self.gamma
        
        loss = focal_term * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            alpha_t = self.alpha.gather(0, targets)
            
            loss = alpha_t * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def compute_class_weights(labels, target_names, device='cpu'):
    """
    Calculates class weights inversely proportional to their frequency.
    
    Args:
        labels (array-like): Dataset labels (can be strings or indices)
        target_names (list): Ordered list of class names
        device (str): Device to put weight tensor on
    
    Returns:
        torch.Tensor: Tensor with weights for each class
    """
    # If labels are strings, convert to numeric indices
    if isinstance(labels[0], str):
        damage_to_idx = {name: i for i, name in enumerate(target_names)}
        numeric_labels = np.array([damage_to_idx[label] for label in labels])
    else:
        numeric_labels = np.array(labels)
    
 
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(numeric_labels), 
        y=numeric_labels
    )
    
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    return weights


def create_loss_function(config, train_df):
    """
    Creates loss function based on configuration.
    Factory function that instantiates the appropriate loss.

    Args:
        config: Configuration object (with attributes loss_function, target_names, etc.)
        train_df (pd.DataFrame): Training DataFrame with 'damage' column
    
    Returns:
        nn.Module: Configured loss function
    """
    loss_name = config.loss_function.lower()

    if loss_name == 'cross_entropy':
        print("-> Using standard Cross-Entropy Loss.")
        return nn.CrossEntropyLoss()

    elif loss_name == 'weighted_cross_entropy':
        print("-> Using Weighted Cross-Entropy Loss with automatically calculated weights.")
        
        weights = compute_class_weights(
            train_df['damage'].values,
            config.target_names,
            config.device
        )
        
        print(f"   Calculated weights for classes {config.target_names}: {weights.cpu().numpy().round(3)}")
        
        return nn.CrossEntropyLoss(weight=weights)

    elif loss_name == 'focal_loss':
        print("-> Using Focal Loss to handle class imbalance.")

        alpha_weights = None
        
        if config.focal_alpha is True:
            print("AUTO mode: Automatic alpha weight calculation...")
            
            alpha_weights = compute_class_weights(
                train_df['damage'].values,
                config.target_names,
                config.device
            )
            
            print(f"   Calculated alpha weights: {alpha_weights.cpu().numpy().round(3)}")
            
        elif config.focal_alpha is False:
            print("   GAMMA-ONLY mode: No alpha weighting, only gamma focusing")
            alpha_weights = None
            
        else:
            print("   CUSTOM mode: Using specified alpha weights...")
            alpha_weights = torch.tensor(config.focal_alpha, dtype=torch.float32).to(config.device)
            print(f"   Custom alpha weights: {alpha_weights.cpu().numpy().round(3)}")
        
        print(f"   Gamma: {config.focal_gamma}")
        
        return FocalLoss(alpha=alpha_weights, gamma=config.focal_gamma, reduction='mean')

    else:
        raise ValueError(f"Loss function '{config.loss_function}' not supported. "
                        "Choose from: 'cross_entropy', 'weighted_cross_entropy', 'focal_loss'.")

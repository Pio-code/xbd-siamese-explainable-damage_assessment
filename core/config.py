"""
Contains:
- TrainingConfig: Configuration for training
- InferenceConfig: Configuration for inference
- SegmentationConfig: Segmentation parameters for XAI (LIME/SHAP)
"""

import os
import datetime
import torch

class TrainingConfig:
    """
    Configuration class that encapsulates all training parameters.
    
    Contains:
    - Training hyperparameters (batch size, epochs, learning rate, dropout)
    - Model configuration (backbone, pretrained, num_classes, hidden_size, use_timm)
    - Loss function configuration (type, class weights, focal loss params)
    - Data augmentation configuration (types and probabilities)
    - Device configuration (CPU/GPU, num_workers, mixed precision)
    """
    def __init__(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.base_data_dir = os.path.join(script_dir, 'patched_dataset_unified')
        self.hdf5_dataset_path = os.path.join(self.base_data_dir, 'xbd_dataset.hdf5')
        self.labels_csv_path = os.path.join(self.base_data_dir, 'labels.csv')
        
        # Create a unique directory name for this experiment
        experiment_name = f"exp_{timestamp}"
        self.results_dir = os.path.join(script_dir, 'results', experiment_name)

        # TRAINING HYPERPARAMETERS
        self.batch_size = 96         
        self.num_epochs = 12         
        self.learning_rate = 1e-4   
        self.dropout_rate = 0.5      
        self.n_splits = 5
        
        # TRAINING STRATEGY 
        self.use_gradual_unfreezing = True   
        self.frozen_epochs = 4       # Phase 1: only classifier_head trained
        self.finetune_learning_rate = 1e-5  # Phase 2: reduced LR

        # MODEL CONFIGURATION
        self.backbone_name = 'convnext_small'  
        # CNN: 'efficientnet_b0', 'efficientnet_b3', 'resnet50', 'convnext_tiny', 'convnext_small'
        # Transformers: 'swin_tiny', 'swin_small'
        self.pretrained = True
        self.num_classes = 4
        self.hidden_size = 512
        
        # BACKBONE LIBRARY: True=timm (ImageNet-22K), False=torchvision (ImageNet-1K)
        self.use_timm = False     # timm: 14M img, 21.8K classes | torchvision: 1.2M img, 1K classes
        
        # LOSS FUNCTION CONFIGURATION
        self.loss_function = 'cross_entropy'  # Options: 'cross_entropy', 'weighted_cross_entropy', 'focal_loss'
        self.class_weights = None  # Automatically calculated for weighted_cross_entropy
        
        # FOCAL LOSS PARAMETERS
        # focal_alpha controls per-class weights in Focal Loss:
        #   True: automatically calculate weights (class balancing)
        #   False: no weights, use only gamma focusing
        #   [w0, w1, w2, w3]: custom weights for each class
        self.focal_alpha = False  
        self.focal_gamma = 2.0   # Gamma focusing parameter (default: 2.0)
        
        # DATA AUGMENTATION CONFIGURATION
        self.enable_data_augmentation = True  
        self.augment_horizontal_flip_prob = 0.5   
        self.augment_vertical_flip_prob = 0.5      
        self.augment_rotation_prob = 0.5          
        self.augment_color_jitter_prob = 0.8      
        self.augment_brightness = 0.20             
        self.augment_contrast = 0.20                
        self.augment_saturation = 0.20            
        self.augment_hue = 0.1                     # Hue variation factor
        
        # DEVICE CONFIGURATION
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 4  # if pickle data was truncated error occurs, restart at 0 then reset
        self.mixed_precision = True  
        
        # CLASS NAMES
        self.target_names = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
    
    def __str__(self):
        """Readable representation of configuration for debugging."""
        augment_status = "ENABLED" if self.enable_data_augmentation else "DISABLED"
        training_strategy = "GRADUAL (Feature Extraction -> Fine-Tuning)" if self.use_gradual_unfreezing else "FULL"
        
        # Backbone library
        backbone_library = "timm (ImageNet-22K: 14M img, 21.8K classes)" if self.use_timm else "torchvision (ImageNet-1K: 1.2M img, 1K classes)"
        
        # Show loss parameters only if relevant
        loss_details = f"{self.loss_function}"
        if self.loss_function == 'weighted_cross_entropy':
            loss_details += f" (weights: {self.class_weights if self.class_weights else 'auto'})"
        elif self.loss_function == 'focal_loss':
            loss_details += f" (alpha: {self.focal_alpha}, gamma: {self.focal_gamma})"
        
        return f"""TrainingConfig:
  Batch Size: {self.batch_size}
  Epochs: {self.num_epochs}
  Learning Rate: {self.learning_rate}
  Dropout Rate: {self.dropout_rate}
  Folds: {self.n_splits}
  num_workers: {self.num_workers}
  Device: {self.device}
  Training Strategy: {training_strategy}
  Frozen Epochs: {self.frozen_epochs if self.use_gradual_unfreezing else 'N/A'}
  Fine-Tune LR: {self.finetune_learning_rate if self.use_gradual_unfreezing else 'N/A'}
  Model: {self.backbone_name} (pretrained={self.pretrained})
  Backbone Library: {backbone_library}
  Loss Function: {loss_details}
  Data Augmentation: {augment_status}
  Data Dir: {self.base_data_dir}
  Results Dir: {self.results_dir}
"""


class InferenceConfig:
    """
    Configuration for inference on test set.
    Must match TrainingConfig to ensure architecture consistency.
    
    Contains:
    - Model configuration (backbone, pretrained, num_classes, hidden_size, use_timm)
    - Data configuration (path to test set)
    """
    def __init__(self):
        # EXPERIMENT CONFIGURATION
        self.experiment_name = 'exp_8_convenext_small'  
        self.backbone_name = 'convnext_small'  # Must match the experiment
        # if it doesn't match, weights are loaded into a different architecture causing errors
        # or if no error, it's still weights trained in a different architecture
        self.hidden_size = 512  # must match the experiment
        
        # BACKBONE LIBRARY: True=timm (ImageNet-22K), False=torchvision (ImageNet-1K)
        self.use_timm = False  # must match the experiment training
        
        # MODEL CONFIGURATION (must match training)
        self.pretrained = True
        self.num_classes = 4
        self.dropout_rate = 0.5
        
        # PATHS
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # TRAINING RESULTS (base results directory)
        self.results_dir = os.path.join(script_dir, 'results')
        # EXPERIMENT DIR (full path to specific experiment)
        self.experiment_dir = os.path.join(self.results_dir, self.experiment_name)
        
        # TEST DATA (path to preprocessed test set)
        self.base_test_dir = os.path.join(script_dir, '02_inference', 'data_prep_test', 'patched_dataset_test')
        self.hdf5_test_dataset_path = os.path.join(self.base_test_dir, 'xbd_test_dataset.hdf5')
        self.labels_test_csv_path = os.path.join(self.base_test_dir, 'labels_test.csv')
        
        # INFERENCE PARAMETERS
        self.batch_size = 32  
        self.num_workers = 4  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # CLASS MAPPING
        self.idx_to_class = {0: 'no-damage', 1: 'minor-damage', 2: 'major-damage', 3: 'destroyed'}
        self.target_names = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']


class SegmentationConfig:
    """
    Configuration for XAI segmentation parameters (LIME and SHAP).
    These parameters control how the image is divided into superpixels.
    """
    def __init__(self):
        self.algorithm = 'felzenszwalb'  # 'felzenszwalb', 'quickshift', 'slic'
        
        # FELZENSZWALB PARAMETERS (graph-based segmentation)
        self.felzenszwalb_scale = 150.0      # Scale parameter: higher values = larger superpixels
        self.felzenszwalb_sigma = 0.5       # Gaussian blur sigma: pre-segmentation smoothing
        self.felzenszwalb_min_size = 50      # Minimum superpixel size (in pixels)
        
        # QUICKSHIFT PARAMETERS
        self.quickshift_kernel_size = 3.0    # Kernel size for mode-seeking
        self.quickshift_max_dist = 100.0     # Maximum distance in color space
        self.quickshift_ratio = 0.90         # Balance space vs color 
        
        # SLIC PARAMETERS (Simple Linear Iterative Clustering)
        self.slic_n_segments = 50           # Target number of superpixels
        self.slic_compactness = 4           # Balance space vs color (higher = more compact)
    
    def get_segmentation_params(self):
        """
        Returns parameters in the format required by lime.wrappers.scikit_image.SegmentationAlgorithm.
        
        SegmentationAlgorithm requires:
        - algo_type (str): first positional argument
        - kwargs: algorithm-specific parameters
        
        Returns:
            tuple: (algo_type, kwargs_dict) for SegmentationAlgorithm(algo_type, kwargs)
        """
        algo_type = self.algorithm
        
        if self.algorithm == 'felzenszwalb':
            kwargs = {
                'scale': self.felzenszwalb_scale,
                'sigma': self.felzenszwalb_sigma,
                'min_size': self.felzenszwalb_min_size
            }
        elif self.algorithm == 'quickshift':
            kwargs = {
                'kernel_size': self.quickshift_kernel_size,
                'max_dist': self.quickshift_max_dist,
                'ratio': self.quickshift_ratio
            }
        elif self.algorithm == 'slic':
            kwargs = {
                'n_segments': self.slic_n_segments,
                'compactness': self.slic_compactness
            }
        else:
            raise ValueError(f"Unrecognized algorithm: {self.algorithm}")
        
        return algo_type, kwargs
    
    def __str__(self):
        """Readable representation for debugging."""
        if self.algorithm == 'felzenszwalb':
            params = f"scale={self.felzenszwalb_scale}, sigma={self.felzenszwalb_sigma}, min_size={self.felzenszwalb_min_size}"
        elif self.algorithm == 'quickshift':
            params = f"kernel={self.quickshift_kernel_size}, max_dist={self.quickshift_max_dist}, ratio={self.quickshift_ratio}"
        else:  # slic
            params = f"n_segments={self.slic_n_segments}, compactness={self.slic_compactness}"
        
        return f"SegmentationConfig: {self.algorithm} ({params})"


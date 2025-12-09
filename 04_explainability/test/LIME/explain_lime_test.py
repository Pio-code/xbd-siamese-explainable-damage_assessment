''' 
Script to analyze TEST SET examples with LIME 
'''

import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import textwrap
from typing import Optional, Dict, Any, Tuple

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHYTON_DIR = os.path.join(SCRIPT_DIR, '..', '..', '..')
if PHYTON_DIR not in sys.path:
    sys.path.insert(0, PHYTON_DIR)

from core.config import TrainingConfig, InferenceConfig, SegmentationConfig
from core.dataset import BuildingDamageDatasetHDF5, data_transforms
from core.models import SiameseNetwork
from core.xai_shared import (
    denormalize_image,
    create_anchor_and_perturb_prediction_function,
    create_lime_superpixel_heatmap,
    get_top_superpixels_by_importance
)



class ExplainLIMETestConfig:
    """
    Configuration for LIME analysis on TEST SET.
    
    Indices are relative to the test set (labels_test.csv).
    """
    # PARAMETRI PRINCIPALI
    EXAMPLE_INDEX: int = 1576  
    
    FOLD_TO_EXPLAIN: int = 1
    
    EXPERIMENT_TO_EXPLAIN: str = "exp_8_convenext_small"  
    
    BACKBONE_NAME: str = 'convnext_small'  # Options: 'efficientnet_b0', 'efficientnet_b3', 'resnet50', 'convnext_tiny', 'convnext_small'
    
   # None = use the automatic path, otherwise specify the full path
    CUSTOM_MODEL_PATH: Optional[str] = None
    
    # Target class for explanation (None = use model prediction)
    # 0=no-damage, 1=minor-damage, 2=major-damage, 3=destroyed
    TARGET_CLASS_INDEX: Optional[int] = None


    # LIME PARAMETERS

    # Number of perturbed samples for LIME 
    NUM_SAMPLES: int = 2000

    # Maximum number of superpixels to show in the visualization
    NUM_FEATURES: int = 100

    USE_STRATIFICATION: bool = True
    
    # VISUAL OPTIONS 
    SHOW_PLOT: bool = True
    FIGURE_WIDTH: float = 18.0
    FIGURE_HEIGHT: float = 15.0
    SAVE_DPI: int = 200
    
    GRID_ROWS: int = 2
    GRID_COLS: int = 3
    
    VERTICAL_SPACING: float = 0.15   
    HORIZONTAL_SPACING: float = 0.10  
    
    LEFT_MARGIN: float = 0.05
    RIGHT_MARGIN: float = 0.95
    TOP_MARGIN: float = 0.92
    BOTTOM_MARGIN: float = 0.05
    
    OUTPUT_DIR: str = "test_single"  
    
    def __init__(self) -> None:
        """Builds the model path dynamically."""
        
        # Calculate base paths (file is in test/LIME/)
        lime_dir = os.path.dirname(os.path.abspath(__file__))  
        test_dir = os.path.dirname(lime_dir)  
        explainability_dir = os.path.dirname(test_dir)  
        phyton_dir = os.path.dirname(explainability_dir) 
        
        self.output_dir = os.path.join(explainability_dir, 'xai_results', self.OUTPUT_DIR)
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.CUSTOM_MODEL_PATH:
            self.MODEL_PATH = self.CUSTOM_MODEL_PATH
            print(f" Using custom model: {self.MODEL_PATH}")
        else:
            self.MODEL_PATH = os.path.join(
                phyton_dir, 'results', self.EXPERIMENT_TO_EXPLAIN,
                f'fold_{self.FOLD_TO_EXPLAIN}', 'models',
                f'best_model_fold_{self.FOLD_TO_EXPLAIN}.pth'
            )
            print(f" Using model FOLD {self.FOLD_TO_EXPLAIN}: {self.EXPERIMENT_TO_EXPLAIN}")


config_explain = ExplainLIMETestConfig()

config = TrainingConfig()  
inference_config = InferenceConfig()  

LABELS_CSV_PATH: str = inference_config.labels_test_csv_path
HDF5_DATASET_PATH: str = inference_config.hdf5_test_dataset_path

MODEL_WEIGHTS_PATH: str = config_explain.MODEL_PATH
EXAMPLE_INDEX: int = config_explain.EXAMPLE_INDEX
TARGET_CLASS_INDEX: Optional[int] = config_explain.TARGET_CLASS_INDEX

print("\n" + "="*80)
print(" TEST SET MODE - LIME Analysis on unseen data")
print("="*80)
print(f"Dataset: {LABELS_CSV_PATH}")
print(f"HDF5: {HDF5_DATASET_PATH}")
print(f"Model: {MODEL_WEIGHTS_PATH}")
print("="*80 + "\n")

if TARGET_CLASS_INDEX is not None and TARGET_CLASS_INDEX >= len(config.target_names):
    print(f"   ERROR: TARGET_CLASS_INDEX ({TARGET_CLASS_INDEX}) is invalid")
    print(f"   Use an index between 0 and {len(config.target_names)-1}")
    print(f"   Available classes: {config.target_names}")
    exit(1)

idx_to_class: Dict[int, str] = {i: name for i, name in enumerate(config.target_names)}

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(MODEL_WEIGHTS_PATH):
    print(f"   ERROR: Model file not found!")
    print(f"   Path searched: {MODEL_WEIGHTS_PATH}")
    exit(1)

if not os.path.exists(LABELS_CSV_PATH):
    print(f"   ERROR: labels_test.csv file not found")
    print(f"   Path searched: {LABELS_CSV_PATH}")
    exit(1)

if not os.path.exists(HDF5_DATASET_PATH):
    print(f"   ERROR: Test HDF5 file not found!")
    print(f"   Path searched: {HDF5_DATASET_PATH}")
    exit(1)

# Initialize model
config.backbone_name = config_explain.BACKBONE_NAME
print(f" Using backbone: {config.backbone_name}")

model = SiameseNetwork(config)  
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
model.to(device)
model.eval()

print(f" Model loaded from: {MODEL_WEIGHTS_PATH}")
print(f" Test dataset loaded from: {LABELS_CSV_PATH}")

all_labels_df: pd.DataFrame = pd.read_csv(LABELS_CSV_PATH)
print(f" Test set: {len(all_labels_df)} samples")

if EXAMPLE_INDEX >= len(all_labels_df):
    print(f"   ERROR: EXAMPLE_INDEX ({EXAMPLE_INDEX}) >= number of samples ({len(all_labels_df)})")
    print(f"   Use an index between 0 and {len(all_labels_df)-1}")
    exit(1)

full_dataset = BuildingDamageDatasetHDF5(hdf5_path=HDF5_DATASET_PATH, transform=None)

img_pre_tensor, img_post_tensor, label_idx = full_dataset[EXAMPLE_INDEX]
label_idx = label_idx.item() if isinstance(label_idx, torch.Tensor) else label_idx
label_name: str = idx_to_class[label_idx]

print(f"\n Analysis of TEST example #{EXAMPLE_INDEX}")

sample_info: pd.Series = all_labels_df.iloc[EXAMPLE_INDEX]
print(f"  Real Label: {label_name} (idx: {label_idx})")

if 'disaster_type' in sample_info:
    print(f"  Tipo di Disastro: {sample_info['disaster_type']}")

# Model prediction
input_tensor_pre: torch.Tensor = img_pre_tensor.unsqueeze(0).to(device)
input_tensor_post: torch.Tensor = img_post_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor_pre, input_tensor_post)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_idx = probabilities.argmax(dim=1).item()
    predicted_name = idx_to_class[predicted_idx]
    
print(f"  Model prediction: {predicted_name} (idx: {predicted_idx})")
print(f"  Confidence: {probabilities.max().item():.4f}")


if TARGET_CLASS_INDEX is None:
    TARGET_CLASS_INDEX = predicted_idx
    print(f" Target class: '{idx_to_class[TARGET_CLASS_INDEX]}' (model prediction)")
else:
    print(f" Target class: '{idx_to_class[TARGET_CLASS_INDEX]}' (user specified)")

print(f"\n Configuring LIME...")

img_pre_viz: np.ndarray = denormalize_image(img_pre_tensor)
img_post_viz: np.ndarray = denormalize_image(img_post_tensor)

img_pre_tensor_original: torch.Tensor = img_pre_tensor.clone().unsqueeze(0).to(device)

explainer = lime_image.LimeImageExplainer()
predict_fn = create_anchor_and_perturb_prediction_function(
    model, img_pre_tensor_original, device, data_transforms
)

seg_config = SegmentationConfig()
algo_type, kwargs = seg_config.get_segmentation_params()
segmentation_fn = SegmentationAlgorithm(algo_type, **kwargs)

print(f"  Generating LIME explanation")
print(f"   Segmentation algorithm: {seg_config.algorithm}")
print(f"   Samples to generate: {config_explain.NUM_SAMPLES}")
print(f"   Stratified sampling: {'ACTIVE' if config_explain.USE_STRATIFICATION else 'INACTIVE'}")

explanation = explainer.explain_instance(
    img_post_viz,
    predict_fn,
    labels=[TARGET_CLASS_INDEX],
    top_labels=1,
    num_features=config_explain.NUM_FEATURES,
    num_samples=config_explain.NUM_SAMPLES,
    segmentation_fn=segmentation_fn,
    hide_color=(0, 0, 0),
    use_stratification=config_explain.USE_STRATIFICATION
)

print(f" LIME explanation successfully generated")

def create_superpixel_visualization(base_image: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Create a visualization of all superpixels with colored borders."""
    superpixel_viz = base_image.copy()
    
    from skimage.segmentation import mark_boundaries
    
    # Use mark_boundaries to highlight superpixel boundaries
    superpixel_viz = mark_boundaries(superpixel_viz, segments, color=(1, 1, 0), mode='thick')
    
    return superpixel_viz


# Prepare visualizations
segments = explanation.segments

# Extract superpixel importance on target class. List of pairs: [(superpixel_id, importance_weight), ...]
feature_importance = explanation.local_exp[TARGET_CLASS_INDEX]
num_superpixels = len(np.unique(segments))

lime_heatmap = create_lime_superpixel_heatmap(img_post_viz, segments, feature_importance)
superpixel_visualization = create_superpixel_visualization(img_post_viz, segments)

diff_image: np.ndarray = np.abs(img_pre_viz - img_post_viz)
diff_image = (diff_image - diff_image.min()) / (diff_image.max() - diff_image.min())

top_superpixels = get_top_superpixels_by_importance(feature_importance)

# VISUALIZATION

fig = plt.figure(figsize=(config_explain.FIGURE_WIDTH, config_explain.FIGURE_HEIGHT))

gs = gridspec.GridSpec(
    config_explain.GRID_ROWS, config_explain.GRID_COLS,
    figure=fig,
    height_ratios=[1.0, 1.2],  # Row 1: images, Row 2: explanations (slightly taller)
    width_ratios=[1.0, 1.0, 1.0],  
    hspace=config_explain.VERTICAL_SPACING,
    wspace=config_explain.HORIZONTAL_SPACING,
    left=config_explain.LEFT_MARGIN,
    right=config_explain.RIGHT_MARGIN,
    top=config_explain.TOP_MARGIN,
    bottom=config_explain.BOTTOM_MARGIN
)

fig.suptitle(f"LIME Analysis - TEST SET Example #{EXAMPLE_INDEX}", fontsize=20, weight='bold')

# ROW 1: IMAGES
ax_pre = fig.add_subplot(gs[0, 0])
ax_pre.imshow(img_pre_viz)
ax_pre.set_title('Pre-Disaster Image', fontsize=13, weight='bold')
ax_pre.axis('off')

ax_post = fig.add_subplot(gs[0, 1])
ax_post.imshow(img_post_viz)
ax_post.set_title('Post-Disaster Image', fontsize=13, weight='bold')
ax_post.axis('off')

ax_diff = fig.add_subplot(gs[0, 2])
ax_diff.imshow(diff_image, cmap='hot')
ax_diff.set_title('Absolute Difference', fontsize=13, weight='bold')
ax_diff.axis('off')

# ROW 2: EXPLANATIONS
ax_segments = fig.add_subplot(gs[1, 0])
ax_segments.imshow(superpixel_visualization)
ax_segments.set_title('Superpixel Segmentation', fontsize=12, weight='bold')
ax_segments.axis('off')

ax_lime = fig.add_subplot(gs[1, 1])
ax_lime.imshow(lime_heatmap)
ax_lime.set_title('LIME Heatmap Post-Disaster', fontsize=12, weight='bold')
ax_lime.axis('off')

# TEXT BLOCK
ax_info = fig.add_subplot(gs[1, 2])
ax_info.axis('off')

post_image_name = os.path.basename(sample_info.get('post_patch_path', 'N/D'))
wrapped_image_name = textwrap.fill(post_image_name, width=35)

positive_count = sum(1 for _, imp in feature_importance if imp > 0)
negative_count = sum(1 for _, imp in feature_importance if imp < 0)

info_text_lines = [
    f" TEST SET ANALYSIS",
    f"TECHNIQUE: LIME",
    f"Model: FOLD {config_explain.FOLD_TO_EXPLAIN}",
    f"Disaster: {sample_info.get('disaster_type', 'N/A')}",
    f"Image: {wrapped_image_name}",
    "",
    f"Real Label: {label_name.upper()}",
    f"Prediction: {predicted_name.upper()}",
    f"Target: '{idx_to_class[TARGET_CLASS_INDEX].upper()}'",
    f"Confidence: {probabilities.max().item():.2%}",
    "",
    f"LIME Configuration:",
    f"  • Algorithm: {seg_config.algorithm}",
    f"  • Superpixels: {num_superpixels}",
    "",
    "Top 3 Superpixels:"
]

# Add top 3 superpixels
for i, (feature_id, importance) in enumerate(top_superpixels[:3]):
    sign = "+" if importance > 0 else ""
    contribution = "PRO" if importance > 0 else "CONTRO"
    info_text_lines.append(f"  {i+1}. SP#{feature_id}: {sign}{importance:.3f} ({contribution})")

info_text = "\n".join(info_text_lines)

ax_info.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=9.5, 
             linespacing=1.3, weight='normal',
             bbox=dict(boxstyle="round,pad=0.8", fc='lightgreen', ec='darkgreen', lw=2, alpha=0.9))

if config_explain.SHOW_PLOT:
    plt.show()

backbone_short = config_explain.BACKBONE_NAME.replace('efficientnet_', 'efficient').replace('_', '')
output_filename = f'test_{EXAMPLE_INDEX}_lime_{backbone_short}_f{config_explain.FOLD_TO_EXPLAIN}.png'
output_path = os.path.join(config_explain.output_dir, output_filename)

fig.savefig(output_path, dpi=config_explain.SAVE_DPI, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print(f"\n LIME analysis saved to: {output_path}")

print(f"\n Configuration used:")
print(f"   • Dataset: TEST SET (unseen)")
print(f"   • XAI Technique: LIME")
print(f"   • Model: FOLD {config_explain.FOLD_TO_EXPLAIN}")
print(f"   • Strategy: Fixed PRE, perturbations only on POST")
print(f"   • Sampling: {'Stratified' if config_explain.USE_STRATIFICATION else 'Standard'}")
print(f"   • Example: TEST #{EXAMPLE_INDEX}")
print(f"   • Target class: {idx_to_class[TARGET_CLASS_INDEX]}")
print(f"   • Segmentation algorithm: {seg_config.algorithm}")
print(f"   • Generated superpixels: {num_superpixels}")
print(f"   • LIME samples: {config_explain.NUM_SAMPLES}")

print(f"\n Top 5 Superpixels by importance:")
for i, (feature_id, importance) in enumerate(top_superpixels[:5]):
    contribution = "Positivo" if importance > 0 else "Negativo"
    print(f"   {i+1}. Superpixel #{feature_id}: {importance:.4f} ({contribution})")

print("\n" + "="*80)
print(" Analysis completed!")
print("="*80)

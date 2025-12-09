'''
Script to generate LIME explanations for ALL 4 CLASSES on TEST SET
Shows what LIME "sees" for each possible damage prediction
'''

import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
)


class ExplainLIMEAllClassesTestConfig:
   
    EXAMPLE_INDEX: int = 31841

    FOLD_TO_EXPLAIN: int = 1  
    EXPERIMENT_TO_EXPLAIN: str = "exp_8_convenext_small" 
    
    BACKBONE_NAME: str = 'convnext_small' 
    
    CUSTOM_MODEL_PATH: Optional[str] = None
    
    NUM_SAMPLES: int = 10000
    NUM_FEATURES: int = 100
    USE_STRATIFICATION: bool = True
    
    SHOW_PLOT: bool = True
    FIGURE_SIZE: tuple = (12, 8)  
    SAVE_DPI: int = 300
    OUTPUT_DIR: str = "test_single"
    
    def __init__(self) -> None:
        """dynamically constructs the model path."""
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        lime_dir = os.path.dirname(os.path.abspath(__file__)) 
        test_dir = os.path.dirname(lime_dir) 
        explainability_dir = os.path.dirname(test_dir)  
        self.output_dir = os.path.join(explainability_dir, 'xai_results', self.OUTPUT_DIR)
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.CUSTOM_MODEL_PATH:
            self.MODEL_PATH = self.CUSTOM_MODEL_PATH
            print(f" Using custom model path: {self.MODEL_PATH}")
        else:
            self.MODEL_PATH = os.path.join(
                script_dir, 'results', self.EXPERIMENT_TO_EXPLAIN,
                f'fold_{self.FOLD_TO_EXPLAIN}', 'models',
                f'best_model_fold_{self.FOLD_TO_EXPLAIN}.pth'
            )
            print(f" Using model FOLD {self.FOLD_TO_EXPLAIN}: {self.EXPERIMENT_TO_EXPLAIN}")
        
        print(f" XAI method: LIME for ALL 4 CLASSES (TEST SET)")


config_explain = ExplainLIMEAllClassesTestConfig()
train_config = TrainingConfig()
inference_config = InferenceConfig()

LABELS_CSV_PATH: str = inference_config.labels_test_csv_path
HDF5_DATASET_PATH: str = inference_config.hdf5_test_dataset_path
MODEL_WEIGHTS_PATH: str = config_explain.MODEL_PATH
EXAMPLE_INDEX: int = config_explain.EXAMPLE_INDEX

print("\n" + "="*80)
print(" TEST SET MODE - LIME Analysis for ALL CLASSES")
print("="*80)
print(f"Dataset: {LABELS_CSV_PATH}")
print(f"Model: {MODEL_WEIGHTS_PATH}")
print("="*80 + "\n")

idx_to_class: Dict[int, str] = {i: name for i, name in enumerate(train_config.target_names)}

# LOAD MODEL AND DATA
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(MODEL_WEIGHTS_PATH):
    print(f"  ERROR: Model file not found!")
    print(f"   Path: {MODEL_WEIGHTS_PATH}")
    exit(1)

if not os.path.exists(LABELS_CSV_PATH):
    print(f"  ERROR: File labels_test.csv not found!")
    print(f"   Path: {LABELS_CSV_PATH}")
    exit(1)

if not os.path.exists(HDF5_DATASET_PATH):
    print(f"  ERROR: HDF5 test file not found!")
    print(f"   Path: {HDF5_DATASET_PATH}")
    exit(1)

train_config.backbone_name = config_explain.BACKBONE_NAME
print(f" Using backbone: {train_config.backbone_name}")

model = SiameseNetwork(train_config)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
model.to(device)
model.eval()

print(f" Model loaded from: {os.path.basename(MODEL_WEIGHTS_PATH)}")

# Load TEST dataset
all_labels_df: pd.DataFrame = pd.read_csv(LABELS_CSV_PATH)
print(f" Test set loaded: {len(all_labels_df)} patches")

if EXAMPLE_INDEX >= len(all_labels_df):
    print(f"  ERROR: EXAMPLE_INDEX ({EXAMPLE_INDEX}) out of range!")
    exit(1)

full_dataset = BuildingDamageDatasetHDF5(hdf5_path=HDF5_DATASET_PATH, transform=None)

# Load example
img_pre_tensor, img_post_tensor, label_idx = full_dataset[EXAMPLE_INDEX]
label_idx = label_idx.item() if isinstance(label_idx, torch.Tensor) else label_idx
label_name: str = idx_to_class[label_idx]

input_tensor_pre: torch.Tensor = img_pre_tensor.unsqueeze(0).to(device)
input_tensor_post: torch.Tensor = img_post_tensor.unsqueeze(0).to(device)

print(f"\n  Analysis of TEST example #{EXAMPLE_INDEX}")

sample_info: pd.Series = all_labels_df.iloc[EXAMPLE_INDEX]
print(f"   Real Label: {label_name} (idx: {label_idx})")
if 'disaster_type' in sample_info:
    print(f"   Disaster Type: {sample_info['disaster_type']}")

with torch.no_grad():
    output = model(input_tensor_pre, input_tensor_post)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_idx = probabilities.argmax(dim=1).item()
    predicted_name = idx_to_class[predicted_idx]
    
print(f"   Prediction: {predicted_name} (confidence: {probabilities.max().item():.2%})")
print(f"\n   Confidences for all classes:")
for class_idx, class_name in idx_to_class.items():
    conf = probabilities[0, class_idx].item()
    marker = "★" if class_idx == predicted_idx else " "
    print(f"    {marker} {class_name}: {conf:.2%}")


print(f"\n LIME Configuration...")

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

print(f" Generating LIME explanations for ALL 4 CLASSES...")
print(f"   Samples per explanation: {config_explain.NUM_SAMPLES}")

explanation = explainer.explain_instance(
    img_post_viz,
    predict_fn,  # returns probabilities for all classes
    labels=[0, 1, 2, 3],  #  gives explanation for the indicated classes (all 4)
    top_labels=4,  #  Returns the top (all 4)
    num_features=config_explain.NUM_FEATURES,
    num_samples=config_explain.NUM_SAMPLES,
    segmentation_fn=segmentation_fn,
    hide_color=(0, 0, 0),
    use_stratification=config_explain.USE_STRATIFICATION
)

print(f" LIME explanations generated for all classes")

segments = explanation.segments
num_superpixels = len(np.unique(segments))

# Function to overlay superpixel boundaries on heatmap
def add_superpixel_boundaries(heatmap: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Adds thin superpixel boundaries on the LIME heatmap."""
    from skimage.segmentation import mark_boundaries
    # Use thin yellow boundaries
    heatmap_with_boundaries = mark_boundaries(heatmap, segments, color=(1, 1, 0), mode='subpixel')
    return heatmap_with_boundaries

class_explanations = {}
for class_idx, class_name in idx_to_class.items():
    feature_importance = explanation.local_exp[class_idx]
    heatmap = create_lime_superpixel_heatmap(img_post_viz, segments, feature_importance)
    
    heatmap_with_boundaries = add_superpixel_boundaries(heatmap, segments)
    
    positive_count = sum(1 for _, imp in feature_importance if imp > 0)
    negative_count = sum(1 for _, imp in feature_importance if imp < 0)
    
    class_explanations[class_name] = {
        'importance': feature_importance,
        'heatmap': heatmap_with_boundaries,
        'positive_count': positive_count,
        'negative_count': negative_count
    }
    
    print(f"   {class_name}: {positive_count} positivi, {negative_count} negativi")


# VISUALIZATION
fig = plt.figure(figsize=config_explain.FIGURE_SIZE)

gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    height_ratios=[1.0, 1.0],  
    width_ratios=[1.0, 1.0, 1.0],  
    hspace=0.15,  
    wspace=0.02,  
    left=0.02,
    right=0.98,
    top=0.92,
    bottom=0.02
)

#fig.suptitle(f"Analisi LIME per TUTTE LE CLASSI - TEST SET #{EXAMPLE_INDEX} (FOLD {config_explain.FOLD_TO_EXPLAIN})", 
#             fontsize=18, weight='bold')

class_names_list = list(idx_to_class.values())

#  ROW 1: PRE/POST IMAGES + CLASS 0 HEATMAP

# [0,0] PRE IMAGE
ax_pre = fig.add_subplot(gs[0, 0])
ax_pre.imshow(img_pre_viz)
ax_pre.set_title('PRE-Disaster Image', fontsize=12, weight='bold', color='darkcyan')
ax_pre.axis('off')

# Helper for visualization style
def get_class_style(c_idx, p_idx, l_idx):
    if c_idx == p_idx and c_idx == l_idx:
        return 'green', 4, " (CORRECT)"
    elif c_idx == p_idx:
        return 'red', 4, " (PREDICTED)"
    elif c_idx == l_idx:
        return 'green', 4, " (REAL)"
    else:
        return 'darkcyan', 2, ""

# [0,1] Class 0: no-damage
class_0_name = class_names_list[0]
ax_class0 = fig.add_subplot(gs[0, 1])
ax_class0.imshow(class_explanations[class_0_name]['heatmap'])
conf_0 = probabilities[0, 0].item()
border_color, border_width, title_suffix = get_class_style(0, predicted_idx, label_idx)
ax_class0.set_title(f'{class_0_name.upper()}{title_suffix}\nConfidence: {conf_0:.1%}', 
                   fontsize=12, weight='bold', color=border_color)
for spine in ax_class0.spines.values():
    spine.set_edgecolor(border_color)
    spine.set_linewidth(border_width)
ax_class0.axis('off')

# [0,2] Classe 1: minor-damage
class_1_name = class_names_list[1]
ax_class1 = fig.add_subplot(gs[0, 2])
ax_class1.imshow(class_explanations[class_1_name]['heatmap'])
conf_1 = probabilities[0, 1].item()
border_color, border_width, title_suffix = get_class_style(1, predicted_idx, label_idx)
ax_class1.set_title(f'{class_1_name.upper()}{title_suffix}\nConfidence: {conf_1:.1%}', 
                   fontsize=12, weight='bold', color=border_color)
for spine in ax_class1.spines.values():
    spine.set_edgecolor(border_color)
    spine.set_linewidth(border_width)
ax_class1.axis('off')

#  ROW 2: POST IMAGE + CLASSES 2-3 HEATMAPS

# [1,0] POST IMAGE
ax_post = fig.add_subplot(gs[1, 0])
# Add superpixel boundaries also to original image
img_post_with_boundaries = add_superpixel_boundaries(img_post_viz, segments)
ax_post.imshow(img_post_with_boundaries)
ax_post.set_title('POST-Disaster Image', fontsize=12, weight='bold', color='darkcyan')
ax_post.axis('off')

# [1,1] Classe 2: major-damage
class_2_name = class_names_list[2]
ax_class2 = fig.add_subplot(gs[1, 1])
ax_class2.imshow(class_explanations[class_2_name]['heatmap'])
conf_2 = probabilities[0, 2].item()
border_color, border_width, title_suffix = get_class_style(2, predicted_idx, label_idx)
ax_class2.set_title(f'{class_2_name.upper()}{title_suffix}\nConfidence: {conf_2:.1%}', 
                   fontsize=12, weight='bold', color=border_color)
for spine in ax_class2.spines.values():
    spine.set_edgecolor(border_color)
    spine.set_linewidth(border_width)
ax_class2.axis('off')

# [1,2] Classe 3: destroyed
class_3_name = class_names_list[3]
ax_class3 = fig.add_subplot(gs[1, 2])
ax_class3.imshow(class_explanations[class_3_name]['heatmap'])
conf_3 = probabilities[0, 3].item()
border_color, border_width, title_suffix = get_class_style(3, predicted_idx, label_idx)
ax_class3.set_title(f'{class_3_name.upper()}{title_suffix}\nConfidence: {conf_3:.1%}', 
                   fontsize=12, weight='bold', color=border_color)
for spine in ax_class3.spines.values():
    spine.set_edgecolor(border_color)
    spine.set_linewidth(border_width)
ax_class3.axis('off')

# Show plot
if config_explain.SHOW_PLOT:
    plt.show()

# Save figure
backbone_short = config_explain.BACKBONE_NAME.replace('efficientnet_', 'efficient').replace('_', '')
output_filename = f'test_{EXAMPLE_INDEX}_lime_ALL_CLASSES_{backbone_short}_f{config_explain.FOLD_TO_EXPLAIN}.png'
output_path = os.path.join(config_explain.output_dir, output_filename)

fig.savefig(output_path, dpi=config_explain.SAVE_DPI, transparent=True)
print(f"\n Visualization saved to: {output_path}")

# Summary
print(f"\n ANALYSIS COMPLETED!")
print(f"   • Technique: LIME for ALL 4 CLASSES")
print(f"   • Dataset: TEST SET (unseen)")
print(f"   • Example: TEST #{EXAMPLE_INDEX}")
print(f"   • Model: FOLD {config_explain.FOLD_TO_EXPLAIN}")
print(f"   • Explanations generated: 4 (one per class)")
print(f"   • Total superpixels: {num_superpixels}")
print(f"   • Predicted class: {predicted_name} (Green=Correct, Red=Incorrect)")
print(f"   • Figure: {config_explain.FIGURE_SIZE} @ {config_explain.SAVE_DPI} DPI")

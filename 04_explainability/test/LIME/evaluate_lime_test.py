''' 
Script to evaluate the quality of LIME explanations on A SINGLE TEST SET example
Uses "Deletion" and "Insertion" metrics with AUC calculation
'''

import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, Dict, Any, Tuple, List
from PIL import Image
from sklearn.metrics import auc

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
    mask_superpixels
)
from core.xai_metrics import compute_lime_deletion_insertion_curves



class LIMEEvaluationTestConfig:
    """Configuration for LIME evaluation on a single TEST SET example."""
    
    EXAMPLE_INDEX: int = 32837 
    
    FOLD_TO_EXPLAIN: int = 1  
    EXPERIMENT_TO_EXPLAIN: str = "exp_8_convenext_small"
    
    BACKBONE_NAME: str = 'convnext_small'
    
    CUSTOM_MODEL_PATH: Optional[str] = None
    

    NUM_SAMPLES: int = 2000  
    NUM_FEATURES: int = 100
    
    USE_STRATIFICATION: bool = True
    

    SHOW_PLOT: bool = True
    FIGURE_WIDTH: float = 16.0
    FIGURE_HEIGHT: float = 12.0
    SAVE_DPI: int = 300
    
    OUTPUT_DIR: str = "test_single"  
    
    def __init__(self):
        """Dynamically builds the model path."""
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        test_dir = os.path.dirname(os.path.abspath(__file__))  
        lime_dir = os.path.dirname(test_dir)  
        explainability_dir = os.path.dirname(lime_dir)  
        self.output_dir = os.path.join(explainability_dir, 'xai_results', self.OUTPUT_DIR)
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.CUSTOM_MODEL_PATH:
            self.MODEL_PATH = self.CUSTOM_MODEL_PATH
        else:
            self.MODEL_PATH = os.path.join(
                script_dir, 'results', self.EXPERIMENT_TO_EXPLAIN,
                f'fold_{self.FOLD_TO_EXPLAIN}', 'models',
                f'best_model_fold_{self.FOLD_TO_EXPLAIN}.pth'
            )
        
        print(f"\n{'='*80}")
        print(f"  LIME EVALUATION - TEST SET (Single Example)")
        print(f"{'='*80}")
        print(f"  Model: FOLD {self.FOLD_TO_EXPLAIN}")
        print(f"  Example: TEST #{self.EXAMPLE_INDEX}")
        print(f"  Stratified sampling: {'ACTIVE' if self.USE_STRATIFICATION else 'INACTIVE'}")
        print(f"{'='*80}\n")


class LIMEEvaluatorTest:
    """Main class for LIME evaluation on test set."""
    
    def __init__(self, config: LIMEEvaluationTestConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Carica modello e dati TEST
        self._load_model_and_data()
        self._setup_lime()
        
    def _load_model_and_data(self):
        """load model and test dataset."""
        # configuration training (for num_classes, target_names, etc.)
        train_config = TrainingConfig()
        
        # configuration inference (for paths test set)
        inference_config = InferenceConfig()
        
        # Override backbone_name
        train_config.backbone_name = self.config.BACKBONE_NAME
        print(f" Using backbone: {train_config.backbone_name}")
        
        # Load model
        self.model = SiameseNetwork(train_config)
        self.model.load_state_dict(torch.load(self.config.MODEL_PATH))
        self.model.to(self.device)
        self.model.eval()
        
        self.full_dataset = BuildingDamageDatasetHDF5(
            hdf5_path=inference_config.hdf5_test_dataset_path,  
            transform=None  
        )
        
        # Load TEST labels for additional info (paths from InferenceConfig)
        self.all_labels_df = pd.read_csv(inference_config.labels_test_csv_path)  
        self.idx_to_class = {i: name for i, name in enumerate(train_config.target_names)}
        
        print(f"Model loaded: {self.config.MODEL_PATH}")
        print(f"Test set loaded: {len(self.all_labels_df)} samples")
        
    def _setup_lime(self):
        """Configure LIME explainer."""
        self.explainer = lime_image.LimeImageExplainer()
        
        seg_config = SegmentationConfig()
        algo_type, kwargs = seg_config.get_segmentation_params()
        self.segmentation_fn = SegmentationAlgorithm(algo_type, **kwargs)
            
        print(f" LIME configured with {seg_config.algorithm}")
    
    def _denormalize_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Wrapper for denormalize_image from core.xai_shared."""
        return denormalize_image(tensor)
    
    def _create_prediction_function(self, pre_image_tensor: torch.Tensor):
        """Wrapper for create_anchor_and_perturb_prediction_function from core.xai_shared."""
        return create_anchor_and_perturb_prediction_function(
            self.model, pre_image_tensor, self.device, data_transforms
        )
    
    def _mask_superpixels(self, image: np.ndarray, segments: np.ndarray, 
                         superpixel_ids: List[int], mask_value: float = 0) -> np.ndarray:
        """Wrapper for mask_superpixels from core.xai_shared."""
        return mask_superpixels(image, segments, superpixel_ids, mask_value)
    
    def evaluate_all_classes_single_example(self, example_index: int) -> Dict[str, Any]:
        """
        Evaluate how deletion and insertion of superpixels influence confidence for ALL classes
        on a SINGLE TEST SET example.
        
        Args:
            example_index: Index of the example to analyze in the test set
            
        Returns:
            Dict with deletion and insertion curves for each class, including AUCs
        """
        print(f"\n Evaluating Deletion and Insertion curves for all classes on TEST example #{example_index}")
        
        # Load example
        img_pre_tensor, img_post_tensor, label_idx = self.full_dataset[example_index]
        label_idx = label_idx.item() if isinstance(label_idx, torch.Tensor) else label_idx
        
        # Prepare tensors for model
        input_tensor_pre = img_pre_tensor.unsqueeze(0).to(self.device)
        input_tensor_post = img_post_tensor.unsqueeze(0).to(self.device)
        
        # Original prediction for all classes
        with torch.no_grad():
            output = self.model(input_tensor_pre, input_tensor_post)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_idx = probabilities.argmax(dim=1).item()
        
        print(f"  Predicted class: {self.idx_to_class[predicted_idx]}")
        print(f"  Original confidences for all classes:")
        for class_idx, class_name in self.idx_to_class.items():
            conf = probabilities[0, class_idx].item()
            print(f"    {class_name}: {conf:.3f}")
        
        # Generate LIME explanation (using predicted class)
        print(f"  Generating LIME explanation...")
        img_post_viz = self._denormalize_image(img_post_tensor)
        predict_fn = self._create_prediction_function(input_tensor_pre)
        
        explanation = self.explainer.explain_instance(
            img_post_viz,
            predict_fn,
            labels=[predicted_idx],
            top_labels=1,
            num_features=self.config.NUM_FEATURES,
            num_samples=self.config.NUM_SAMPLES,
            segmentation_fn=self.segmentation_fn,
            hide_color=(0, 0, 0),
            use_stratification=self.config.USE_STRATIFICATION
        )
        
        segments = explanation.segments
        feature_importance = explanation.local_exp[predicted_idx]
        
        # Denormalize POST image for curves
        img_post_viz = self._denormalize_image(img_post_tensor)
        
        print(f"  Computing DELETION and INSERTION curves for all classes...")
        results_curves = compute_lime_deletion_insertion_curves(
            model=self.model,
            pre_tensor=input_tensor_pre,
            post_tensor=input_tensor_post,
            post_image_viz=img_post_viz,
            segments=segments,
            feature_importance=feature_importance,
            target_class=predicted_idx,
            device=self.device,
            idx_to_class=self.idx_to_class,
            mask_value=0.0
        )
        
        print(f"\n  AUC for all classes:")
        for class_name, auc_values in results_curves['auc_scores'].items():
            print(f"    {class_name}:")
            print(f"      Deletion AUC: {auc_values['deletion_auc']:.4f}")
            print(f"      Insertion AUC: {auc_values['insertion_auc']:.4f}")
        
        results = {
            'example_index': example_index,
            'predicted_class': self.idx_to_class[predicted_idx],
            'true_class': self.idx_to_class[label_idx],
            'num_superpixels': len(feature_importance),
            'feature_importance': feature_importance,
            'segments': segments,
            'post_image_viz': img_post_viz,
            'deletion_curves': results_curves['deletion_curves'],
            'insertion_curves': results_curves['insertion_curves'],
            'auc_scores': results_curves['auc_scores']
        }
        
        print(f"\n✓ Valutazione completata!")
        return results
    
    def _create_lime_superpixel_heatmap(self, base_image: np.ndarray, segments: np.ndarray, 
                                       feature_importance: list, alpha: float = 0.7) -> np.ndarray:
        """Wrapper per create_lime_superpixel_heatmap da core.xai_shared."""
        return create_lime_superpixel_heatmap(base_image, segments, feature_importance, alpha)

    def visualize_deletion_insertion_results(self, results: Dict[str, Any]) -> str:
        """Visualizza i risultati della valutazione deletion e insertion con AUC scores."""
        
        fig = plt.figure(figsize=(self.config.FIGURE_WIDTH, self.config.FIGURE_HEIGHT))
        
        gs = gridspec.GridSpec(
            2, 3,
            figure=fig,
            height_ratios=[1.0, 1.0],
            width_ratios=[1.0, 1.0, 0.8],
            hspace=0.20,
            wspace=0.15,
            left=0.05,
            right=0.98,
            top=0.92,
            bottom=0.08
        )
        
        example_idx = results['example_index']
        fig.suptitle(f'Valutazione LIME TEST SET: Deletion & Insertion - Esempio #{example_idx} (FOLD {self.config.FOLD_TO_EXPLAIN})', 
                    fontsize=16, weight='bold')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        predicted_class = results['predicted_class']
        
        #  [0,0] DELETION - PREDICTED CLASS ONLY 
        ax_del_pred = fig.add_subplot(gs[0, 0])
        curve_data = results['deletion_curves'][predicted_class]
        deletion_pct = curve_data['percentages'] * 100
        confidences = curve_data['confidences']
        deletion_auc = results['auc_scores'][predicted_class]['deletion_auc']
        
        ax_del_pred.plot(deletion_pct, confidences, color='red', linewidth=3, 
                        linestyle='-', marker='o', markersize=6, 
                        label=f'{predicted_class} (AUC={deletion_auc:.3f})')
        ax_del_pred.set_xlabel('% Superpixel Rimossi', fontsize=11, weight='bold')
        ax_del_pred.set_ylabel('Confidenza', fontsize=11, weight='bold')
        ax_del_pred.set_title(f'Deletion - Classe Predetta\n({predicted_class.upper()})', fontsize=12, weight='bold')
        ax_del_pred.grid(True, alpha=0.3)
        ax_del_pred.set_ylim(0, 1)
        ax_del_pred.legend(fontsize=9, loc='upper right')
        
        #  [0,1] DELETION - ALL CLASSES 
        ax_del_all = fig.add_subplot(gs[0, 1])
        for i, (class_name, curve_data) in enumerate(results['deletion_curves'].items()):
            color = colors[i % len(colors)]
            deletion_pct = curve_data['percentages'] * 100
            confidences = curve_data['confidences']
            deletion_auc = results['auc_scores'][class_name]['deletion_auc']
            
            linestyle = '-' if class_name == predicted_class else '--'
            linewidth = 3 if class_name == predicted_class else 2
            
            ax_del_all.plot(deletion_pct, confidences, color=color, linewidth=linewidth, 
                           linestyle=linestyle, marker='o', markersize=4, 
                           label=f'{class_name} ({deletion_auc:.3f})')
        
        ax_del_all.set_xlabel('% Superpixel Rimossi', fontsize=11, weight='bold')
        ax_del_all.set_ylabel('Confidenza', fontsize=11, weight='bold')
        ax_del_all.set_title('Deletion - Tutte le Classi\n(AUC in legenda)', fontsize=12, weight='bold')
        ax_del_all.grid(True, alpha=0.3)
        ax_del_all.set_ylim(0, 1)
        ax_del_all.legend(fontsize=8, loc='upper right')
        
        #  [0,2] LIME HEATMAP 
        ax_lime = fig.add_subplot(gs[0, 2])
        lime_heatmap = self._create_lime_superpixel_heatmap(
            results['post_image_viz'], 
            results['segments'], 
            results['feature_importance']
        )
        ax_lime.imshow(lime_heatmap)
        ax_lime.set_title('LIME Heatmap Post-Disastro\n(Verde=Positivo, Rosso=Negativo)', fontsize=12, weight='bold')
        ax_lime.axis('off')
        
        #  [1,0] INSERTION - PREDICTED CLASS ONLY 
        ax_ins_pred = fig.add_subplot(gs[1, 0])
        curve_data = results['insertion_curves'][predicted_class]
        insertion_pct = curve_data['percentages'] * 100
        confidences = curve_data['confidences']
        insertion_auc = results['auc_scores'][predicted_class]['insertion_auc']
        
        ax_ins_pred.plot(insertion_pct, confidences, color='blue', linewidth=3, 
                        linestyle='-', marker='s', markersize=6, 
                        label=f'{predicted_class} (AUC={insertion_auc:.3f})')
        ax_ins_pred.set_xlabel('% Superpixel Mantenuti', fontsize=11, weight='bold')
        ax_ins_pred.set_ylabel('Confidence', fontsize=11, weight='bold')
        ax_ins_pred.set_title(f'Insertion - Predicted Class\n({predicted_class.upper()})', fontsize=12, weight='bold')
        ax_ins_pred.grid(True, alpha=0.3)
        ax_ins_pred.set_ylim(0, 1)
        ax_ins_pred.legend(fontsize=9, loc='upper right')
        
        #  [1,1] INSERTION - ALL CLASSES 
        ax_ins_all = fig.add_subplot(gs[1, 1])
        for i, (class_name, curve_data) in enumerate(results['insertion_curves'].items()):
            color = colors[i % len(colors)]
            insertion_pct = curve_data['percentages'] * 100
            confidences = curve_data['confidences']
            insertion_auc = results['auc_scores'][class_name]['insertion_auc']
            
            linestyle = '-' if class_name == predicted_class else '--'
            linewidth = 3 if class_name == predicted_class else 2
            
            ax_ins_all.plot(insertion_pct, confidences, color=color, linewidth=linewidth, 
                           linestyle=linestyle, marker='s', markersize=4, 
                           label=f'{class_name} ({insertion_auc:.3f})')
        
        ax_ins_all.set_xlabel('% Superpixel Maintained', fontsize=11, weight='bold')
        ax_ins_all.set_ylabel('Confidence', fontsize=11, weight='bold')
        ax_ins_all.set_title('Insertion - All Classes\n(AUC in legend)', fontsize=12, weight='bold')
        ax_ins_all.grid(True, alpha=0.3)
        ax_ins_all.set_ylim(0, 1)
        ax_ins_all.legend(fontsize=8, loc='upper right')
        
        #  [1,2] REPORT WITH AUC 
        ax_report = fig.add_subplot(gs[1, 2])
        ax_report.axis('off')
        
        try:
            sample_info = self.all_labels_df.iloc[example_idx]
            disaster_type = sample_info.get('disaster_type', 'N/D')
        except:
            disaster_type = 'N/D'
        
        feature_importance = results['feature_importance']
        positive_superpixels = sum(1 for _, importance in feature_importance if importance > 0)
        negative_superpixels = sum(1 for _, importance in feature_importance if importance < 0)
        
        # Extract AUC for the predicted class
        pred_del_auc = results['auc_scores'][predicted_class]['deletion_auc']
        pred_ins_auc = results['auc_scores'][predicted_class]['insertion_auc']
        
        report_text = f""" TEST SET ANALYSIS
Example #{example_idx}

INFORMATION:
• Disaster: {disaster_type}
• Model: FOLD {self.config.FOLD_TO_EXPLAIN}
• Total Superpixels: {results['num_superpixels']}
• Positive Superpixels: {positive_superpixels}
• Negative Superpixels: {negative_superpixels}

CLASSIFICATION:
• True: {results['true_class'].upper()}
• Predicted: {predicted_class.upper()}
AUC SCORES (Predicted Class):
• Deletion AUC: {pred_del_auc:.4f}
• Insertion AUC: {pred_ins_auc:.4f}

LIME ALGORITHM:
• Segmentation: {SegmentationConfig().algorithm}
• Samples: {self.config.NUM_SAMPLES}
• Features: {self.config.NUM_FEATURES}"""
        
        ax_report.text(0.05, 0.5, report_text, ha='left', va='center', fontsize=9,
                      linespacing=1.2, weight='normal',
                      bbox=dict(boxstyle="round,pad=0.5", fc='lightblue', ec='darkblue', 
                               lw=2, alpha=0.9))
        

        backbone_short = self.config.BACKBONE_NAME.replace('efficientnet_', 'efficient').replace('_', '')
        output_filename = f'test_{example_idx}_lime_eval_{backbone_short}_f{self.config.FOLD_TO_EXPLAIN}.png'
        output_path = os.path.join(self.config.output_dir, output_filename)
        
        fig.savefig(output_path, dpi=self.config.SAVE_DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        if self.config.SHOW_PLOT:
            plt.show()
        
        print(f"\n Visualization saved: {output_path}")
        return output_path


def main():
    """Main function for executing evaluation on a single TEST example."""
    config = LIMEEvaluationTestConfig()
    
    # Create evaluator
    evaluator = LIMEEvaluatorTest(config)
    
    # Complete evaluation
    print("=== DELETION & INSERTION EVALUATION FOR ALL CLASSES (TEST SET) ===")
    results = evaluator.evaluate_all_classes_single_example(config.EXAMPLE_INDEX)
    
    # Display results
    evaluator.visualize_deletion_insertion_results(results)
    
    print("\n" + "="*80)
    print(" Evaluation completed!")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = main()

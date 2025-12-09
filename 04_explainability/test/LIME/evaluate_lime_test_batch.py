'''
Script to evaluate the quality of LIME explanations on the entire test set
or on a stratified batch
Calculates average AUC for Deletion and Insertion with aggregate statistics
'''

import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Tuple, List
from PIL import Image
from sklearn.metrics import auc
from scipy.interpolate import interp1d
from tqdm import tqdm
import json

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
    mask_superpixels
)
from core.xai_metrics import compute_lime_deletion_insertion_curves


class LIMEBatchEvaluationConfig:
    
    SAMPLES_PER_CLASS: int = 400  # (stratified by disaster type)
    
    ONLY_CORRECT_PREDICTIONS: bool = True  
    
    RANDOM_SEED: int = 42  
    
    FOLD_TO_EXPLAIN: int = 1  #1  best for convnext_small, 3 for convnext_tiny, 4 for resnet50
    EXPERIMENT_TO_EXPLAIN: str = "exp_8_convenext_small" 
    
    BACKBONE_NAME: str = 'convnext_small' # Options: 'efficientnet_b0', 'efficientnet_b3', 'resnet50', 'convnext_tiny', 'convnext_small'
    
    CUSTOM_MODEL_PATH: Optional[str] = None

    NUM_SAMPLES: int = 500  # Reduced compared to single example (2000)
    NUM_FEATURES: int = 100
    USE_STRATIFICATION: bool = True
    
    SAVE_PROGRESS_EVERY: int = 200  
    
    SAVE_CURVES: bool = True  # Save raw curves for each example 
    GENERATE_AVERAGED_CURVES: bool = True  
    
    def __init__(self):
        """Dynamically builds the model path and output directory."""
        # Go up to phyton/ (file is in test/LIME/, need 4 dirname)
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        backbone_short = self.BACKBONE_NAME.replace('_', '')  # convnext_small -> convnextsmall
        output_dir_name = f"test_stratified_{self.SAMPLES_PER_CLASS}perclass_{backbone_short}"
        
        lime_dir = os.path.dirname(os.path.abspath(__file__))  # LIME/
        test_dir = os.path.dirname(lime_dir)  # test/
        explainability_dir = os.path.dirname(test_dir)  # 04_explainability/
        self.output_dir = os.path.join(explainability_dir, 'xai_results', output_dir_name)
        os.makedirs(self.output_dir, exist_ok=True)
        

        if self.CUSTOM_MODEL_PATH:
            self.MODEL_PATH = self.CUSTOM_MODEL_PATH
        else:
            self.MODEL_PATH = os.path.join(
                script_dir, 'results', self.EXPERIMENT_TO_EXPLAIN,
                f'fold_{self.FOLD_TO_EXPLAIN}', 'models',
                f'best_model_fold_{self.FOLD_TO_EXPLAIN}.pth'
            )
        
        filter_msg = " (only CORRECT PREDICTIONS)" if self.ONLY_CORRECT_PREDICTIONS else ""
        
        print(f"\n{'='*80}")
        print(f"  LIME EVALUATION - STRATIFIED SAMPLING")
        print(f"{'='*80}")
        print(f"  Model: {self.BACKBONE_NAME} (FOLD {self.FOLD_TO_EXPLAIN})")
        print(f"  Samples per class: {self.SAMPLES_PER_CLASS} (stratified by disaster type)")
        print(f"  Filter: {filter_msg if self.ONLY_CORRECT_PREDICTIONS else 'ALL predictions'}")
        print(f"  LIME samples: {self.NUM_SAMPLES}")
        print(f"  Save curves: {'YES' if self.SAVE_CURVES else 'NO (AUC only)'}")
        print(f"  Output: {self.output_dir}")
        print(f"{'='*80}\n")


class LIMEBatchEvaluator:
    """Main class for batch LIME evaluation on test set."""
    
    def __init__(self, config: LIMEBatchEvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f" Device: {self.device}")
        
        self._load_model_and_data()
        self._setup_lime()
        
        self._select_samples()
        
    def _load_model_and_data(self):
        """Load model and TEST dataset."""
        train_config = TrainingConfig()
        train_config.backbone_name = self.config.BACKBONE_NAME
        print(f" Using backbone: {train_config.backbone_name}")
        
        inference_config = InferenceConfig()
        
        self.model = SiameseNetwork(train_config)
        self.model.load_state_dict(torch.load(self.config.MODEL_PATH)) 
        self.model.to(self.device)
        self.model.eval()
        
        self.full_dataset = BuildingDamageDatasetHDF5(
            hdf5_path=inference_config.hdf5_test_dataset_path,
            transform=None 
        )
        
        self.all_labels_df = pd.read_csv(inference_config.labels_test_csv_path)
        self.idx_to_class = {i: name for i, name in enumerate(train_config.target_names)}
        self.class_to_idx = {name: i for i, name in self.idx_to_class.items()}
        
        print(f" Model loaded: {self.config.MODEL_PATH}")
        print(f" Total test set: {len(self.all_labels_df)} samples")
        
    def _setup_lime(self):
        """Configure LIME explainer."""
        self.explainer = lime_image.LimeImageExplainer()
        
        seg_config = SegmentationConfig()
        algo_type, kwargs = seg_config.get_segmentation_params()
        self.segmentation_fn = SegmentationAlgorithm(algo_type, **kwargs)

        print(f" LIME configured with {seg_config.algorithm}")
    
    def _stratified_sampling_all_classes(self) -> List[int]:
        """
        Stratified sampling by disaster type on ALL classes.
        Maintains original disaster type proportions for each class.
        
        Returns:
            List of stratified sampled indices
        """
        np.random.seed(self.config.RANDOM_SEED)
        
        all_sampled_indices = []
        
        print(f"\n Stratified sampling ({self.config.SAMPLES_PER_CLASS} per class):")
        print(f"   Strategy: Maintain disaster type proportions for each class\n")
        
        for class_name in self.idx_to_class.values():
            class_mask = self.all_labels_df['damage'] == class_name
            class_indices = self.all_labels_df[class_mask].index.tolist()
            
            disaster_groups = {}
            for idx in class_indices:
                disaster = str(self.all_labels_df.iloc[idx]['disaster_type']).lower()
                if disaster not in disaster_groups:
                    disaster_groups[disaster] = []
                disaster_groups[disaster].append(idx)
            
            total_available = len(class_indices)
            n_samples = min(self.config.SAMPLES_PER_CLASS, total_available)
            
            print(f"   {class_name} (target: {n_samples} samples):")
            
            sampled = []
            for disaster, disaster_indices in disaster_groups.items():
                proportion = len(disaster_indices) / total_available
                target_n = int(np.round(proportion * n_samples))
                available = len(disaster_indices)
                
                if target_n == 0:
                    continue
                
                take_n = min(target_n, available)
                selected = np.random.choice(disaster_indices, size=take_n, replace=False).tolist()
                sampled.extend(selected)
                
                print(f" {disaster}: {len(selected)} samples ({proportion*100:.1f}% of total)")
            
            # Adjust any rounding differences
            if len(sampled) < n_samples:
                remaining = n_samples - len(sampled)
                unsampled = [idx for idx in class_indices if idx not in sampled]
                if len(unsampled) > 0:
                    additional = np.random.choice(unsampled, size=min(remaining, len(unsampled)), replace=False).tolist()
                    sampled.extend(additional)
            elif len(sampled) > n_samples:
                np.random.shuffle(sampled)
                sampled = sampled[:n_samples]
            
            all_sampled_indices.extend(sampled)
            print(f" Total: {len(sampled)} samples\n")

        print(f" Sampling completed: {len(all_sampled_indices)} total samples")
        print(f" Estimated time: ~{len(all_sampled_indices) * 30 / 3600:.1f} hours\n")

        return all_sampled_indices
    
    def _select_samples(self):
        """Select samples to analyze with stratified strategy."""
        self.sample_indices = self._stratified_sampling_all_classes()
    
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
    
    def evaluate_single_example(self, example_index: int) -> Dict[str, Any]:
        """
        Evaluate a single example and calculate AUC for the predicted class.
        Simplified version for batch processing (predicted class only).
        
        Returns:
            Dict with results, or None if the sample should be skipped
        """
        img_pre_tensor, img_post_tensor, label_idx = self.full_dataset[example_index]
        label_idx = label_idx.item() if isinstance(label_idx, torch.Tensor) else label_idx
        
        disaster_type = str(self.all_labels_df.iloc[example_index]['disaster_type'])
        
        input_tensor_pre = img_pre_tensor.unsqueeze(0).to(self.device)
        input_tensor_post = img_post_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor_pre, input_tensor_post)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_idx = probabilities.argmax(dim=1).item()
            original_confidence = probabilities[0, predicted_idx].item()
        
        if self.config.ONLY_CORRECT_PREDICTIONS and predicted_idx != label_idx:
            return None
        
        img_post_viz = self._denormalize_image(img_post_tensor)
        predict_fn = self._create_prediction_function(input_tensor_pre)
        
        try:
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
        except Exception as e:
            print(f"   LIME error on example #{example_index}: {str(e)}")
            return None
        
        segments = explanation.segments
        
        if predicted_idx not in explanation.local_exp:
            print(f"   LIME did not return explanations for predicted class {predicted_idx} on example #{example_index}")
            print(f"    Available classes: {list(explanation.local_exp.keys())}")
            return None
            
        feature_importance = explanation.local_exp[predicted_idx]
        
        img_post_viz = self._denormalize_image(img_post_tensor)
        
        # (calculates for all classes but then extract only predicted class)
        try:
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
        except ValueError as e:
            # No positive superpixels
            print(f"\n  {str(e)} for example #{example_index}")
            return None
        
        predicted_class_name = self.idx_to_class[predicted_idx]
        deletion_auc = results_curves['auc_scores'][predicted_class_name]['deletion_auc']
        insertion_auc = results_curves['auc_scores'][predicted_class_name]['insertion_auc']
        
        result = {
            'example_index': example_index,
            'true_class': self.idx_to_class[label_idx],
            'predicted_class': predicted_class_name,
            'disaster_type': disaster_type,
            'original_confidence': original_confidence,
            'deletion_auc': deletion_auc,
            'insertion_auc': insertion_auc,
            'num_superpixels': len(feature_importance),
            'num_positive_superpixels': results_curves['num_positive_superpixels']
        }
        
        if self.config.SAVE_CURVES:
            result['deletion_curve'] = {
                'percentages': results_curves['deletion_curves'][predicted_class_name]['percentages'].tolist(),
                'confidences': results_curves['deletion_curves'][predicted_class_name]['confidences']
            }
            result['insertion_curve'] = {
                'percentages': results_curves['insertion_curves'][predicted_class_name]['percentages'].tolist(),
                'confidences': results_curves['insertion_curves'][predicted_class_name]['confidences']
            }
        
        return result
    
    def evaluate_batch(self) -> Dict[str, Any]:
        """Evaluate all selected samples and calculate aggregate statistics."""
        print(f"\n{'='*80}")
        print(f"  START BATCH EVALUATION")
        print(f"{'='*80}\n")
        
        all_results = []
        failed_examples = []
        skipped_examples = []  # Incorrect predictions (if filter active)
        
        pbar = tqdm(self.sample_indices, desc="LIME Evaluation", unit="examples")
        
        for idx, example_idx in enumerate(pbar):
            result = self.evaluate_single_example(example_idx)
            
            if result is None:
                # Skipped sample (incorrect prediction with filter active)
                skipped_examples.append(example_idx)
            elif result is not None:
                all_results.append(result)
                
                if len(all_results) > 0:
                    avg_del_auc = np.mean([r['deletion_auc'] for r in all_results])
                    avg_ins_auc = np.mean([r['insertion_auc'] for r in all_results])
                    pbar.set_postfix({
                        'Avg Del AUC': f'{avg_del_auc:.3f}',
                        'Avg Ins AUC': f'{avg_ins_auc:.3f}'
                    })
            else:
                failed_examples.append(example_idx)
            
            if (idx + 1) % self.config.SAVE_PROGRESS_EVERY == 0:
                self._save_intermediate_results(all_results, idx + 1)
        
        pbar.close()
        
        print(f"\n Evaluation completed!")
        print(f"  Processed examples: {len(all_results)}")
        if self.config.ONLY_CORRECT_PREDICTIONS and len(skipped_examples) > 0:
            print(f"  Skipped examples (incorrect predictions): {len(skipped_examples)}")
        print(f"  Failed examples: {len(failed_examples)}")
        
        aggregate_stats = self._compute_aggregate_statistics(all_results)
        
        self._save_final_results(all_results, aggregate_stats, failed_examples, skipped_examples)
        
        self._generate_visualizations(all_results, aggregate_stats)
        
        return {
            'individual_results': all_results,
            'aggregate_stats': aggregate_stats,
            'failed_examples': failed_examples,
            'skipped_examples': skipped_examples
        }
    
    def _compute_aggregate_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate statistics on results."""
        print(f"\n Calculating aggregate statistics...")
        
        df = pd.DataFrame(results)
        
        stats = {
            'overall': {
                'deletion_auc_mean': df['deletion_auc'].mean(),
                'deletion_auc_std': df['deletion_auc'].std(),
                'deletion_auc_median': df['deletion_auc'].median(),
                'insertion_auc_mean': df['insertion_auc'].mean(),
                'insertion_auc_std': df['insertion_auc'].std(),
                'insertion_auc_median': df['insertion_auc'].median(),
                'num_samples': len(df)
            },
            'by_class': {},
            'by_correctness': {}
        }
        
        for class_name in self.idx_to_class.values():
            class_df = df[df['predicted_class'] == class_name]
            if len(class_df) > 0:
                stats['by_class'][class_name] = {
                    'deletion_auc_mean': class_df['deletion_auc'].mean(),
                    'deletion_auc_std': class_df['deletion_auc'].std(),
                    'insertion_auc_mean': class_df['insertion_auc'].mean(),
                    'insertion_auc_std': class_df['insertion_auc'].std(),
                    'num_samples': len(class_df),
                    'avg_confidence': class_df['original_confidence'].mean()
                }
        
        # Statistics by prediction correctness
        # Map class names to indices for comparison
        true_class_indices = df['true_class'].map(self.class_to_idx)
        predicted_class_indices = df['predicted_class'].map(self.class_to_idx)
        df['correct'] = true_class_indices == predicted_class_indices
        
        correct_df = df[df['correct']]
        incorrect_df = df[~df['correct']]
        
        if len(correct_df) > 0:
            stats['by_correctness']['correct'] = {
                'deletion_auc_mean': correct_df['deletion_auc'].mean(),
                'insertion_auc_mean': correct_df['insertion_auc'].mean(),
                'num_samples': len(correct_df)
            }
        
        if len(incorrect_df) > 0:
            stats['by_correctness']['incorrect'] = {
                'deletion_auc_mean': incorrect_df['deletion_auc'].mean(),
                'insertion_auc_mean': incorrect_df['insertion_auc'].mean(),
                'num_samples': len(incorrect_df)
            }
        
        print(f"\n{'='*80}")
        print(f"  AGGREGATE STATISTICS")
        print(f"{'='*80}")
        print(f"\nOVERALL ({stats['overall']['num_samples']} samples):")
        print(f"  Deletion AUC:  {stats['overall']['deletion_auc_mean']:.4f} ± {stats['overall']['deletion_auc_std']:.4f}")
        print(f"  Insertion AUC: {stats['overall']['insertion_auc_mean']:.4f} ± {stats['overall']['insertion_auc_std']:.4f}")
        
        print(f"\nPER CLASS:")
        for class_name, class_stats in stats['by_class'].items():
            print(f"  {class_name} (n={class_stats['num_samples']}):")
            print(f"    Del AUC: {class_stats['deletion_auc_mean']:.4f} ± {class_stats['deletion_auc_std']:.4f}")
            print(f"    Ins AUC: {class_stats['insertion_auc_mean']:.4f} ± {class_stats['insertion_auc_std']:.4f}")
        
        print(f"\nBY CORRECTNESS:")
        if 'correct' in stats['by_correctness']:
            print(f"  Correct predictions (n={stats['by_correctness']['correct']['num_samples']}):")
            print(f"    Del AUC: {stats['by_correctness']['correct']['deletion_auc_mean']:.4f}")
            print(f"    Ins AUC: {stats['by_correctness']['correct']['insertion_auc_mean']:.4f}")
        if 'incorrect' in stats['by_correctness']:
            print(f"  Incorrect predictions (n={stats['by_correctness']['incorrect']['num_samples']}):")
            print(f"    Del AUC: {stats['by_correctness']['incorrect']['deletion_auc_mean']:.4f}")
            print(f"    Ins AUC: {stats['by_correctness']['incorrect']['insertion_auc_mean']:.4f}")
        
        print(f"{'='*80}\n")
        
        return stats
    
    def _compute_averaged_curves(self, results: List[Dict]) -> Optional[Dict]:
        """
        Interpolate and average all curves on common grid.
        
        Returns:
            Dict with averaged curves and std, or None if SAVE_CURVES=False
        """
        if not self.config.SAVE_CURVES:
            print(f"\n    SAVE_CURVES=False → Skip averaged curves calculation")
            print(f"      To enable: config.SAVE_CURVES = True")
            return None
        
        if not results or 'deletion_curve' not in results[0]:
            print(f"\n    No curves saved in results → Skip averaged curves")
            return None
        
        print(f"\n Calculating interpolated averaged curves...")
        
        # Common grid: 100 points from 0% to 100%
        common_x = np.linspace(0, 1, 100)
        
        # Collect all interpolated curves
        deletion_interp = []
        insertion_interp = []
        
        for result in results:
            try:
                del_x = np.array(result['deletion_curve']['percentages'])
                del_y = np.array(result['deletion_curve']['confidences'])
                
                # Linear interpolation with bounds handling
                f_del = interp1d(del_x, del_y, kind='linear', bounds_error=False, 
                                fill_value=(del_y[0], del_y[-1]))
                deletion_interp.append(f_del(common_x))
                
                # Interpolate insertion curve
                ins_x = np.array(result['insertion_curve']['percentages'])
                ins_y = np.array(result['insertion_curve']['confidences'])
                
                f_ins = interp1d(ins_x, ins_y, kind='linear', bounds_error=False,
                                fill_value=(ins_y[0], ins_y[-1]))
                insertion_interp.append(f_ins(common_x))
                
            except Exception as e:
                print(f"    Interpolation error example #{result['example_index']}: {str(e)}")
                continue
        
        if len(deletion_interp) == 0:
            print(f"   No curves successfully interpolated")
            return None
        
        # Convert to numpy array for statistical calculations
        deletion_interp = np.array(deletion_interp)
        insertion_interp = np.array(insertion_interp)
        
        print(f" Interpolated {len(deletion_interp)} curves")
        
        return {
            'common_percentages': common_x,
            'deletion_mean': np.mean(deletion_interp, axis=0),
            'deletion_std': np.std(deletion_interp, axis=0),
            'deletion_q25': np.percentile(deletion_interp, 25, axis=0),
            'deletion_q75': np.percentile(deletion_interp, 75, axis=0),
            'insertion_mean': np.mean(insertion_interp, axis=0),
            'insertion_std': np.std(insertion_interp, axis=0),
            'insertion_q25': np.percentile(insertion_interp, 25, axis=0),
            'insertion_q75': np.percentile(insertion_interp, 75, axis=0),
            'num_curves': len(deletion_interp)
        }
    
    def _save_intermediate_results(self, results: List[Dict], num_processed: int):
        """Save intermediate results."""
        temp_path = os.path.join(self.config.output_dir, f'intermediate_{num_processed}.json')
        with open(temp_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n Progress saved: {temp_path}")
    
    def _save_final_results(self, results: List[Dict], stats: Dict, failed: List[int], skipped: List[int] = None):
        """Save final results."""
        if skipped is None:
            skipped = []
            
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.config.output_dir, 'individual_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n Individual results saved: {csv_path}")
        
        # Save aggregate statistics as JSON
        stats_path = os.path.join(self.config.output_dir, 'aggregate_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f" Aggregate statistics saved: {stats_path}")
        
        # Save configuration
        config_dict = {
            'samples_per_class': self.config.SAMPLES_PER_CLASS,
            'stratified_by_disaster': True,
            'only_correct_predictions': self.config.ONLY_CORRECT_PREDICTIONS,
            'fold': self.config.FOLD_TO_EXPLAIN,
            'experiment': self.config.EXPERIMENT_TO_EXPLAIN,
            'backbone': self.config.BACKBONE_NAME,
            'lime_samples': self.config.NUM_SAMPLES,
            'segmentation': SegmentationConfig().algorithm,
            'stratification': self.config.USE_STRATIFICATION,
            'random_seed': self.config.RANDOM_SEED,
            'failed_examples': failed,
            'skipped_examples': skipped,
            'total_samples': len(results)
        }
        config_path = os.path.join(self.config.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"✓ Configuration saved: {config_path}")
    
    def _generate_visualizations(self, results: List[Dict], stats: Dict):
        """Generate result visualizations."""
        print(f"\n Generating visualizations...")
        
        df = pd.DataFrame(results)
        
        # === FIGURE 1: AUC Distribution ===
        fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig1.suptitle('AUC Scores Distribution - Test Set', fontsize=16, weight='bold')
        
        # Deletion AUC - histogram
        axes[0, 0].hist(df['deletion_auc'], bins=30, color='red', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(stats['overall']['deletion_auc_mean'], color='darkred', 
                          linestyle='--', linewidth=2, label=f"Mean: {stats['overall']['deletion_auc_mean']:.3f}")
        axes[0, 0].set_xlabel('Deletion AUC', fontsize=11, weight='bold')
        axes[0, 0].set_ylabel('Frequency', fontsize=11, weight='bold')
        axes[0, 0].set_title('Deletion AUC Distribution', fontsize=12, weight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Insertion AUC - histogram
        axes[0, 1].hist(df['insertion_auc'], bins=30, color='blue', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(stats['overall']['insertion_auc_mean'], color='darkblue', 
                          linestyle='--', linewidth=2, label=f"Mean: {stats['overall']['insertion_auc_mean']:.3f}")
        axes[0, 1].set_xlabel('Insertion AUC', fontsize=11, weight='bold')
        axes[0, 1].set_ylabel('Frequency', fontsize=11, weight='bold')
        axes[0, 1].set_title('Insertion AUC Distribution', fontsize=12, weight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC per class - Deletion
        class_names = list(stats['by_class'].keys())
        del_means = [stats['by_class'][c]['deletion_auc_mean'] for c in class_names]
        del_stds = [stats['by_class'][c]['deletion_auc_std'] for c in class_names]
        
        x_pos = np.arange(len(class_names))
        axes[1, 0].bar(x_pos, del_means, yerr=del_stds, color='red', alpha=0.7, 
                      capsize=5, edgecolor='black')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Deletion AUC', fontsize=11, weight='bold')
        axes[1, 0].set_title('Deletion AUC per Class', fontsize=12, weight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # AUC per class - Insertion
        ins_means = [stats['by_class'][c]['insertion_auc_mean'] for c in class_names]
        ins_stds = [stats['by_class'][c]['insertion_auc_std'] for c in class_names]
        
        axes[1, 1].bar(x_pos, ins_means, yerr=ins_stds, color='blue', alpha=0.7, 
                      capsize=5, edgecolor='black')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(class_names, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Insertion AUC', fontsize=11, weight='bold')
        axes[1, 1].set_title('Insertion AUC per Class', fontsize=12, weight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig1_path = os.path.join(self.config.output_dir, 'auc_distributions.png')
        fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"  ✓ Salvata: {fig1_path}")
        
        # === FIGURE 2: Scatter plot and comparisons ===
        fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig2.suptitle('Comparative AUC Scores Analysis', fontsize=16, weight='bold')
        
        # Scatter Deletion vs Insertion
        colors_map = {'no-damage': 'green', 'minor-damage': 'yellow', 
                     'major-damage': 'orange', 'destroyed': 'red'}
        
        for class_name in class_names:
            class_df = df[df['predicted_class'] == class_name]
            axes[0].scatter(class_df['deletion_auc'], class_df['insertion_auc'], 
                          color=colors_map.get(class_name, 'gray'), 
                          label=class_name, alpha=0.6, s=50)
        
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
        axes[0].set_xlabel('Deletion AUC', fontsize=11, weight='bold')
        axes[0].set_ylabel('Insertion AUC', fontsize=11, weight='bold')
        axes[0].set_title('Deletion vs Insertion AUC', fontsize=12, weight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot comparison
        data_to_plot = [df['deletion_auc'], df['insertion_auc']]
        box = axes[1].boxplot(data_to_plot, labels=['Deletion', 'Insertion'], 
                             patch_artist=True, showmeans=True)
        box['boxes'][0].set_facecolor('red')
        box['boxes'][0].set_alpha(0.5)
        box['boxes'][1].set_facecolor('blue')
        box['boxes'][1].set_alpha(0.5)
        
        axes[1].set_ylabel('AUC Score', fontsize=11, weight='bold')
        axes[1].set_title('Deletion vs Insertion Comparison', fontsize=12, weight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig2_path = os.path.join(self.config.output_dir, 'auc_comparisons.png')
        fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"  ✓ Salvata: {fig2_path}")
        
        #  FIGURE 3: Averaged Deletion/Insertion Curves 
        if self.config.GENERATE_AVERAGED_CURVES:
            avg_curves = self._compute_averaged_curves(results)
            
            if avg_curves is not None:
                fig3, axes = plt.subplots(1, 2, figsize=(14, 6))
                fig3.suptitle(f'Averaged Deletion/Insertion Curves - Test Set (n={avg_curves["num_curves"]})', 
                             fontsize=16, weight='bold')
                
                x_pct = avg_curves['common_percentages'] * 100
                
                # Averaged deletion curve with ±std band
                axes[0].plot(x_pct, avg_curves['deletion_mean'], 'r-', linewidth=3, label='Mean')
                axes[0].fill_between(x_pct,
                                    avg_curves['deletion_mean'] - avg_curves['deletion_std'],
                                    avg_curves['deletion_mean'] + avg_curves['deletion_std'],
                                    color='red', alpha=0.2, label='±1 std')
                # Add quartiles for robustness
                axes[0].fill_between(x_pct,
                                    avg_curves['deletion_q25'],
                                    avg_curves['deletion_q75'],
                                    color='darkred', alpha=0.1, label='Q25-Q75')
                axes[0].set_xlabel('% Superpixels Removed', fontsize=11, weight='bold')
                axes[0].set_ylabel('Confidence', fontsize=11, weight='bold')
                axes[0].set_title('Averaged Deletion Curve', fontsize=12, weight='bold')
                axes[0].legend(fontsize=9)
                axes[0].grid(True, alpha=0.3)
                axes[0].set_ylim(0, 1)
                axes[0].set_xlim(0, 100)
                
                # Averaged insertion curve with ±std band
                axes[1].plot(x_pct, avg_curves['insertion_mean'], 'b-', linewidth=3, label='Mean')
                axes[1].fill_between(x_pct,
                                    avg_curves['insertion_mean'] - avg_curves['insertion_std'],
                                    avg_curves['insertion_mean'] + avg_curves['insertion_std'],
                                    color='blue', alpha=0.2, label='±1 std')
                # Add quartiles
                axes[1].fill_between(x_pct,
                                    avg_curves['insertion_q25'],
                                    avg_curves['insertion_q75'],
                                    color='darkblue', alpha=0.1, label='Q25-Q75')
                axes[1].set_xlabel('% Superpixels Added', fontsize=11, weight='bold')
                axes[1].set_ylabel('Confidence', fontsize=11, weight='bold')
                axes[1].set_title('Averaged Insertion Curve', fontsize=12, weight='bold')
                axes[1].legend(fontsize=9)
                axes[1].grid(True, alpha=0.3)
                axes[1].set_ylim(0, 1)
                axes[1].set_xlim(0, 100)
                
                plt.tight_layout()
                fig3_path = os.path.join(self.config.output_dir, 'averaged_curves.png')
                fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
                plt.close(fig3)
                print(f" Saved: {fig3_path}")
            else:
                print(f" Skip Figure 3 (averaged curves not available)")
        else:
            print(f" Skip Figure 3 (GENERATE_AVERAGED_CURVES=False)")
        
        print(f" Visualizations completed!\n")


def main():
    """Main function for batch evaluation execution."""
    config = LIMEBatchEvaluationConfig()
    
    # Create evaluator
    evaluator = LIMEBatchEvaluator(config)
    
    # Execute batch evaluation
    results = evaluator.evaluate_batch()
    
    print(f"\n{'='*80}")
    print(f" Batch Evaluation Completed")
    print(f"{'='*80}")
    print(f"  Results saved to: {config.output_dir}")
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    results = main()

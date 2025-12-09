"""
Utilities for inference on trained models.

This module contains functions used to:
- Run inference with single models or ensembles
- Save predictions in CSV format
- Manage DataLoader for test set
"""

import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm


def run_inference_single_model(model, test_loader, device):
    """
    Runs inference with a single model.
    
    Args:
        model: Model in eval mode
        test_loader: DataLoader with test data
        device: Device (cuda/cpu)
    
    Returns:
        tuple: (predictions, true_labels, logits)
            - predictions: np.array of predicted indices
            - true_labels: np.array of true labels
            - logits: np.array of raw model output
    """
    all_predictions = []
    all_labels = []
    all_logits = []
    
    model.eval()
    
    with torch.no_grad():
        for img_pre, img_post, labels in tqdm(test_loader, desc="Inference"):
            img_pre = img_pre.to(device)
            img_post = img_post.to(device)
            
            # Forward pass
            outputs = model(img_pre, img_post)
            
            # Get predictions (class with maximum probability)
            _, predicted = torch.max(outputs, 1)
            
      
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_logits.append(outputs.cpu().numpy())
    
    all_logits = np.vstack(all_logits)
    
    return np.array(all_predictions), np.array(all_labels), all_logits


def run_inference_ensemble(model_paths, test_loader, inference_config):
    """
    Runs inference with model ensemble (probability averaging).
    
    Args:
        model_paths: List of paths to models to use for ensemble
        test_loader: DataLoader with test data
        inference_config: InferenceConfig object
    
    Returns:
        tuple: (ensemble_predictions, labels, ensemble_logits)
            - ensemble_predictions: np.array of final predictions
            - labels: np.array of true labels
            - ensemble_logits: np.array of averaged logits
    """
    from core.models import load_model_from_checkpoint  
    
    print(f"\n{'='*60}")
    print(f"INFERENCE WITH ENSEMBLE OF {len(model_paths)} MODELS")
    print(f"{'='*60}")
    
    all_logits_per_model = []
    labels = None
    
    for idx, model_path in enumerate(model_paths):
        print(f"\n--- Model {idx+1}/{len(model_paths)} ---")
        print(f"Loading from: {model_path}")
        
        model = load_model_from_checkpoint(model_path, inference_config)
        print("Model loaded successfully")
        
        _, batch_labels, logits = run_inference_single_model(
            model, test_loader, inference_config.device
        ) # _ indicates we don't use single predictions, only logits to
        # calculate ensemble at the end
        
        all_logits_per_model.append(logits)
        
        if labels is None:
            labels = batch_labels
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    print("\n Computing ensemble (logits averaging)")
    ensemble_logits = np.mean(all_logits_per_model, axis=0)
    
    ensemble_predictions = np.argmax(ensemble_logits, axis=1)
    
    return ensemble_predictions, labels, ensemble_logits



def save_predictions_to_csv(predictions, true_labels, labels_df, output_path, 
                            inference_config, logits=None):
    """
    Saves predictions to a CSV file with disaster_type and probabilities.
    
    Data order is preserved through the pipeline:
    HDF5 -> DataLoader -> predictions -> CSV

    Args:
        predictions: Numpy array with predictions (class indices)
        true_labels: Numpy array with true labels
        labels_df: DataFrame with metadata (damage, disaster_type, etc.)
        output_path: Path where to save the CSV
        inference_config: InferenceConfig object for class mapping
        logits: (Optional) Numpy array with logits to calculate probabilities
    
    Returns:
        pd.DataFrame: DataFrame with saved predictions
    """
    predicted_classes = [inference_config.idx_to_class[pred] for pred in predictions]
    true_classes = [inference_config.idx_to_class[label] for label in true_labels]
    
    results_df = pd.DataFrame({
        'true_label': true_classes,
        'predicted_label': predicted_classes,
        'true_label_idx': true_labels,
        'predicted_label_idx': predictions
    })
    
    if 'disaster_type' in labels_df.columns:
        results_df['disaster_type'] = labels_df['disaster_type'].values[:len(predictions)]
    
    path_column = None
    if 'pre_path' in labels_df.columns:
        path_column = 'pre_path'
    elif 'pre_patch_path' in labels_df.columns:
        path_column = 'pre_patch_path'
    elif 'post_path' in labels_df.columns:
        path_column = 'post_path'
    elif 'post_patch_path' in labels_df.columns:
        path_column = 'post_patch_path'
    
    if path_column:
        def extract_event_name(path):
            """Extracts event name from path (e.g. 'guatemala-volcano' from full path)"""
            if pd.isna(path):
                return 'unknown'
            # Path format: 'images\guatemala-volcano_00000003_88703461-a33d-4327-9244-a0d4e2242ede_pre.png'
            # or: 'tier3/images/hurricane-florence_00000001_pre_disaster.png'
            try:
                # Take filename (last part after / or \)
                filename = path.replace('\\', '/').split('/')[-1]
                # Remove extension
                filename_no_ext = filename.rsplit('.', 1)[0]
                # Extract event name (everything before first '_' preceding a number)
                parts = filename_no_ext.split('_')
                # Search for pattern: event-name_00000XXX_...
                # The event_name is everything before the first group of 8 digits
                event_parts = []
                for part in parts:
                    if len(part) == 8 and part.isdigit():
                        # Found sequential number, stop here
                        break
                    event_parts.append(part)
                
                event_name = '_'.join(event_parts) if event_parts else 'unknown'
                return event_name
            except:
                return 'unknown'
        
        results_df['event_name'] = labels_df[path_column].apply(extract_event_name).values[:len(predictions)]
    
    # Add probabilities if available
    if logits is not None:
        probabilities = torch.nn.functional.softmax(torch.from_numpy(logits), dim=1).numpy()
        for idx, class_name in inference_config.idx_to_class.items():
            results_df[f'prob_{class_name}'] = probabilities[:, idx]
    
    results_df.to_csv(output_path, index=False)
    print(f"\n Predictions saved: {output_path}")
    
    return results_df


def print_inference_summary(predictions, true_labels, results_df):
    """
    Prints a summary of predictions.
    
    Args:
        predictions: Numpy array with predictions
        true_labels: Numpy array with true labels
        results_df: DataFrame with complete results
    """
    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY")
    print("=" * 60)
    
    accuracy = (predictions == true_labels).mean() * 100
    print(f" Accuracy on test set: {accuracy:.4f}%")
    print(f" Total predictions: {len(predictions)}")
    
    print("\n Prediction distribution:")
    print(results_df['predicted_label'].value_counts())
    
    print("\n True label distribution:")
    print(results_df['true_label'].value_counts())
    
    print("=" * 60)
    print(f"\n For a detailed evaluation, run 'evaluate_model.py'")


def find_model_paths(experiment_dir, num_folds=5):
    """
    Finds paths to best models for each fold.
    
    Args:
        experiment_dir: Experiment directory
        num_folds: Number of folds to search for (default: 5)
    
    Returns:
        list: List of paths to found models
    """
    model_paths = []
    
    for fold_num in range(num_folds):
        possible_paths = [
            os.path.join(experiment_dir, f'fold_{fold_num}', 'models', f'best_model_fold_{fold_num}.pth'),
            os.path.join(experiment_dir, f'fold_{fold_num}', 'models', 'best_model.pth')
        ]
        
        for model_path in possible_paths:
            if os.path.exists(model_path):
                model_paths.append(model_path)
                break
    
    return model_paths

"""
Inference using the ensemble of 5 models saved during training.
"""

import os
import sys
import pandas as pd
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHYTON_DIR = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, PHYTON_DIR)

from core.dataset import BuildingDamageDatasetHDF5
from core.config import InferenceConfig
from core.inference_utils import (
    run_inference_ensemble,
    save_predictions_to_csv,
    print_inference_summary,
    find_model_paths
)


# INFERENCE CONFIGURATION
# To change experiment/model, modify values in InferenceConfig
config = InferenceConfig()

# Test data paths 
HDF5_TEST_PATH = config.hdf5_test_dataset_path
LABELS_TEST_PATH = config.labels_test_csv_path


def main():
    """
    Main function for ensemble inference.
    """
    print("=" * 60)
    print("INFERENCE ON TEST SET - ENSEMBLE")
    print("=" * 60)
    print(f"Experiment: {config.experiment_name}")
    print(f"Device: {config.device}")
    print(f"Inference batch size: {config.batch_size}")
    
    if not os.path.exists(HDF5_TEST_PATH):
        print(f"\n ERROR: HDF5 test file not found: {HDF5_TEST_PATH}")
        return
    
    if not os.path.exists(LABELS_TEST_PATH):
        print(f"\n ERROR: Test labels file not found: {LABELS_TEST_PATH}")
        return
    
    print(f"\n Loading test catalog: {LABELS_TEST_PATH}")
    labels_df = pd.read_csv(LABELS_TEST_PATH)
    print(f" Number of samples in test set: {len(labels_df)}")
    
    # Create dataset and dataloader (without augmentation)
    print("\n Creating test dataset...")
    test_dataset = BuildingDamageDatasetHDF5(HDF5_TEST_PATH, transform=None)
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,  
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Find trained models
    experiment_dir = config.experiment_dir
    
    if not os.path.exists(experiment_dir):
        print(f"\n ERROR: Experiment directory not found: {experiment_dir}")
        return
    
    # Search for models from each fold using core.inference_utils
    model_paths = find_model_paths(experiment_dir, num_folds=5)
    
    if not model_paths:
        print(f"\n ERROR: No models found in {experiment_dir}")
        return
    
    print(f"\n Found {len(model_paths)} models to use for ensemble")
    
    # Run ensemble inference using core.inference_utils
    predictions, true_labels, logits = run_inference_ensemble(
        model_paths, test_loader, config
    )
    
    # Save predictions in experiment directory
    predictions_dir = os.path.join(experiment_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    output_path = os.path.join(predictions_dir, f'predictions_{config.experiment_name}.csv')
    
    # Ensure labels_df has the same number of rows as predictions
    # (dataloader might have excluded some samples if batch_size doesn't divide evenly)
    labels_df_aligned = labels_df.iloc[:len(predictions)].reset_index(drop=True)
    
    # Save using core.inference_utils
    results_df = save_predictions_to_csv(
        predictions, true_labels, labels_df_aligned, output_path, config, logits
    )
    
    print_inference_summary(predictions, true_labels, results_df)


if __name__ == '__main__':
    main()

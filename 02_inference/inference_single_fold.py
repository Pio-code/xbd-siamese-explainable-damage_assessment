"""
Uses a single model from a specific cross-validation fold
"""

import os
import sys
import pandas as pd
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHYTON_DIR = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, PHYTON_DIR)

from core.dataset import BuildingDamageDatasetHDF5
from core.models import load_model_from_checkpoint
from core.config import InferenceConfig
from core.inference_utils import (
    run_inference_single_model,
    save_predictions_to_csv,
    print_inference_summary
)


# INFERENCE CONFIGURATION
# To change experiment, modify values in InferenceConfig

FOLD_NUMBER = 4  # Fold number to test (0-4)

config = InferenceConfig()

# Test data paths 
HDF5_TEST_PATH = config.hdf5_test_dataset_path
LABELS_TEST_PATH = config.labels_test_csv_path


def main():
    """Main function for inference with single fold model."""
    
    print("=" * 60)
    print("INFERENCE ON TEST SET - SINGLE FOLD MODEL")
    print("=" * 60)
    print(f"Experiment: {config.experiment_name}")
    print(f"Fold: {FOLD_NUMBER}")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    
    if not os.path.exists(HDF5_TEST_PATH):
        print(f"\nERROR: HDF5 test file not found: {HDF5_TEST_PATH}")
        return
    
    if not os.path.exists(LABELS_TEST_PATH):
        print(f"\nERROR: Test labels file not found: {LABELS_TEST_PATH}")
        return

    print(f"\nLoading test catalog: {LABELS_TEST_PATH}")
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
    
    # Find model for specified fold
    model_path = os.path.join(
        config.experiment_dir,
        f'fold_{FOLD_NUMBER}',
        'models',
        f'best_model_fold_{FOLD_NUMBER}.pth'
    )
    
    if not os.path.exists(model_path):
        print(f"\n ERROR: Model not found: {model_path}")
        print(f"Make sure experiment '{config.experiment_name}' and fold {FOLD_NUMBER} exist.")
        return
    
    print(f"\n Model found: {model_path}")
    
    # Load model using core.models
    print(f"Loading model from: {model_path}")
    model = load_model_from_checkpoint(model_path, config)
    print(" Model loaded successfully")
    
    # Run inference using core.inference_utils
    predictions, true_labels, logits = run_inference_single_model(
        model, test_loader, config.device
    )
    
    # Save predictions in fold directory
    fold_dir = os.path.join(config.experiment_dir, f'fold_{FOLD_NUMBER}')
    predictions_dir = os.path.join(fold_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    output_path = os.path.join(predictions_dir, f'predictions_fold_{FOLD_NUMBER}_test.csv')
    
    # Save using core.inference_utils
    results_df = save_predictions_to_csv(
        predictions, true_labels, labels_df, output_path, config, logits
    )
    
    print_inference_summary(predictions, true_labels, results_df)
    

if __name__ == '__main__':
    main()

"""
K-Fold Cross-Validation training script for xBD damage assessment.

BACKBONE SUPPORT:
  CNN: ResNet50, EfficientNet (B0/B3), ConvNeXt (Tiny/Small)
  Transformers: Swin Transformer (Tiny/Small)

CUSTOMIZATION:
- Edit `core/config.py` to choose backbone and training parameters.

OPERATIONAL NOTES
Tested with: 32GB RAM, GPU NVIDIA GeForce RTX 4070 8GB, SSD NVMe 1TB
Impractical for Swin Small model due to limited GPU memory.
If mixed precision false, also impractical for Swin Tiny and ConvNeXt Small
"""

import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHYTON_DIR = os.path.join(SCRIPT_DIR, '..')
if PHYTON_DIR not in sys.path:
    sys.path.insert(0, PHYTON_DIR)

from core.config import TrainingConfig
from core.dataset import (create_training_dataset, create_validation_dataset)
from core.models import SiameseNetwork  
from core.losses import create_loss_function
from core.utils import Logger, setup_directories, setup_fold_directories



def collate_fn_skip_corrupted(batch):
    """
    Intercepts the batch before assembly and discards 
    any sample that caused an error during loading    
    """
    original_size = len(batch)
    # Filter 'batch' list, keeping only non-None elements
    batch = list(filter(lambda x: x is not None, batch))
    
    if len(batch) < original_size:
        print(f"Removed {original_size - len(batch)} corrupted images in this batch.")
        
    return torch.utils.data.dataloader.default_collate(batch)

def freeze_backbone(model):
    """
    Freezes all backbone (feature extractor) parameters.
    
    During Feature Extraction, only the head (classifier_head) is trained,
    while the pre-trained backbone remains fixed.
    
    Args:
        model: SiameseNetwork with backbone and classifier_head
    """
    for param in model.backbone.parameters():
        param.requires_grad = False
    print(" Backbone FROZEN - Only classifier_head will be trained")


def unfreeze_backbone(model):
    """
    Unfreezes all backbone parameters for fine-tuning.
    """
    for param in model.backbone.parameters():
        param.requires_grad = True
    print(" Backbone unfrozen - Entire model will be trained")


def create_data_loaders(train_df, val_df, config):
    """
    DataLoader organizes samples into batches and provides them iteratively to the model.
    Uses Subset to filter samples for the current fold.
    
    ARCHITECTURE:
    - HDF5 contains all samples 
    - train_df/val_df contain only indices for the current fold 
    - Subset filters HDF5 to access only fold samples (avoids data duplication)  
      
    Args:
        train_df (DataFrame): Training sample indices for the current fold
        val_df (DataFrame): Validation sample indices for the current fold
        config (TrainingConfig): Configuration with augmentation parameters and batch size
        
    Returns:
        tuple: (train_loader, val_loader, train_subset, val_subset)
    """
    if config.enable_data_augmentation:
        print("Training dataset with Data Augmentation")
        train_dataset = create_training_dataset(
            hdf5_path=config.hdf5_dataset_path,
            enable_augmentation=True,
            config=config  
        )
    else:
        print("Training dataset without Data Augmentation")
        train_dataset = create_training_dataset(config.hdf5_dataset_path, enable_augmentation=False)
    
    print("Validation dataset (no augmentation)")
    val_dataset = create_validation_dataset(config.hdf5_dataset_path)
    
    train_indices = train_df.index.tolist()
    val_indices = val_df.index.tolist()
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(dataset=train_subset, 
                          batch_size=config.batch_size,
                          shuffle=True,  
                          num_workers=config.num_workers,  
                          pin_memory=True,  # Speeds up CPU->GPU transfer
                          collate_fn=collate_fn_skip_corrupted)

    val_loader = DataLoader(dataset=val_subset,
                        batch_size=config.batch_size,
                        shuffle=False,  
                        num_workers=config.num_workers,
                        pin_memory=True,
                        collate_fn=collate_fn_skip_corrupted)  
    
    return train_loader, val_loader, train_subset, val_subset

def train_one_epoch(model, train_loader, optimizer, criterion, config, fold_k, epoch, scaler=None):
    """
    Executes a single training epoch with forward/backward pass.
    
    Returns:
        float: Average epoch loss
    """
    model.train()  
    train_loss = 0.0
    train_loop = tqdm(train_loader, desc=f"Fold {fold_k} Epoch {epoch+1}/{config.num_epochs} [Training]")
    
    for batch_idx, (pre_images, post_images, labels) in enumerate(train_loop):
        pre_images = pre_images.to(config.device, non_blocking=True)
        post_images = post_images.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True)
        
        optimizer.zero_grad()  
        
        # Mixed precision: fp16 calculations, fp32 weights
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(pre_images, post_images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()  
            scaler.step(optimizer)  
            scaler.update()  #
        else:
            # Standard training (fp32)
            outputs = model(pre_images, post_images)
            loss = criterion(outputs, labels)
            loss.backward()  
            optimizer.step()  
        
        train_loss += loss.item()
        if batch_idx % 10 == 0:
            train_loop.set_postfix(loss=loss.item())
    
    return train_loss / len(train_loader)

def validate_model(model, val_loader, criterion, config, fold_k, epoch, val_df):
    """
    Evaluates the model on validation set and collects detailed metrics.
    
    Returns:
        tuple: (avg_val_loss, val_f1, all_labels, all_preds, detailed_results_df)
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    detailed_results = []
    val_loop = tqdm(val_loader, desc=f"Fold {fold_k} Epoch {epoch+1}/{config.num_epochs} [Validation]")
    
    val_indices = val_df.index.tolist()
    
    with torch.no_grad():
        batch_idx = 0
        for pre_images, post_images, labels in val_loop:
            pre_images, post_images, labels = pre_images.to(config.device, non_blocking=True), post_images.to(config.device, non_blocking=True), labels.to(config.device, non_blocking=True)
            
            # Use mixed precision for validation too
            if config.mixed_precision and config.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(pre_images, post_images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(pre_images, post_images)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            batch_preds = preds.cpu().numpy()
            batch_labels = labels.cpu().numpy()
            batch_probs = probabilities.cpu().numpy()
            
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
            
            for i in range(len(batch_labels)):
                loader_idx = batch_idx * val_loader.batch_size + i
                
                if loader_idx < len(val_indices):
                    real_df_idx = val_indices[loader_idx]  
                    
                    detailed_results.append({
                        'sample_index': loader_idx,
                        'true_label': batch_labels[i].item(),
                        'predicted_label': batch_preds[i].item(),
                        'prob_no_damage': batch_probs[i][0].item(),
                        'prob_minor_damage': batch_probs[i][1].item(),
                        'prob_major_damage': batch_probs[i][2].item(),
                        'prob_destroyed': batch_probs[i][3].item(),
                        'disaster_type': val_df.loc[real_df_idx, 'disaster_type']
                    })
            batch_idx += 1
    
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_val_loss = val_loss / len(val_loader)
    
    detailed_results_df = pd.DataFrame(detailed_results)
    
    return avg_val_loss, val_f1, all_labels, all_preds, detailed_results_df

def save_loss_plot(train_losses, val_losses, plots_dir, fold_k, config):
    """
    Generates and saves loss curves plot to monitor overfitting.
    
    Args:
        train_losses: Loss per epoch (training)
        val_losses: Loss per epoch (validation)
        plots_dir: Output directory
        fold_k: Current fold number
        config: Configuration (for num_epochs)
    """
    plt.figure(figsize=(10, 6))  
    
    plt.plot(range(1, config.num_epochs + 1), train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(range(1, config.num_epochs + 1), val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - Fold {fold_k}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loss_plot_path = os.path.join(plots_dir, f'loss_curves_fold_{fold_k}.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Free memory (avoid figure accumulation)
    print(f"-> Loss plot saved: {loss_plot_path}")

def save_confusion_matrix(model, val_loader, plots_dir, fold_k, config):
    """
    Generates confusion matrix to analyze model classification errors.

    Args:
        model: Trained neural network
        val_loader: Validation DataLoader
        plots_dir: Output directory
        fold_k: Current fold number
        config: Configuration (for target_names)
    """
    model.eval()
    
    final_preds = []
    final_labels = []
    with torch.no_grad():
        for pre_images, post_images, labels in val_loader:
            pre_images, post_images, labels = pre_images.to(config.device, non_blocking=True), post_images.to(config.device, non_blocking=True), labels.to(config.device, non_blocking=True)
            
            if hasattr(torch.amp, 'autocast'):
                with torch.amp.autocast('cuda'):
                    outputs = model(pre_images, post_images)
            else:
                outputs = model(pre_images, post_images)
            _, preds = torch.max(outputs, 1)
            final_preds.extend(preds.cpu().numpy())
            final_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(final_labels, final_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=config.target_names,
                yticklabels=config.target_names)
    
    plt.title(f'Confusion Matrix - Fold {fold_k}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    cm_plot_path = os.path.join(plots_dir, f'confusion_matrix_fold_{fold_k}.png')
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Confusion Matrix saved: {cm_plot_path}")


def save_disaster_confusion_matrices(detailed_df, plots_dir, fold_k, config):
    """
    Creates and saves a separate confusion matrix for each disaster type.
    
    Args:
        detailed_df (DataFrame): DataFrame with detailed results (true, predicted, disaster_type).
        plots_dir (str): Directory to save plots.
        fold_k (int): Current fold number.
        config (TrainingConfig): Configuration object.
    """
    disaster_types = detailed_df['disaster_type'].unique()
    print(f"-> Generating Confusion Matrix by disaster type: {list(disaster_types)}")

    for disaster in disaster_types:
        disaster_df = detailed_df[detailed_df['disaster_type'] == disaster]
        
        true_labels = disaster_df['true_label']
        predicted_labels = disaster_df['predicted_label']
        
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Greens', 
                    xticklabels=config.target_names,
                    yticklabels=config.target_names)
        
        plt.title(f'Confusion Matrix - Fold {fold_k} - Disaster: {disaster}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        safe_disaster_name = disaster.replace(" ", "_").lower()
        cm_plot_path = os.path.join(plots_dir, f'confusion_matrix_fold_{fold_k}_{safe_disaster_name}.png')
        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Confusion Matrix for '{disaster}' saved: {cm_plot_path}")

def save_disaster_reports_csv(detailed_df, reports_dir, fold_k):
    """
    Saves a detailed CSV report for each disaster type.
    
    Args:
        detailed_df (DataFrame): Complete DataFrame with results.
        reports_dir (str): Directory to save CSV reports.
        fold_k (int): Current fold number.
    """
    disaster_types = detailed_df['disaster_type'].unique()
    print(f"-> Generating CSV Report by disaster type: {list(disaster_types)}")

    for disaster in disaster_types:
        disaster_df = detailed_df[detailed_df['disaster_type'] == disaster]
        
        safe_disaster_name = disaster.replace(" ", "_").lower()
        report_path = os.path.join(reports_dir, f'validation_report_fold_{fold_k}_{safe_disaster_name}.csv')
        disaster_df.to_csv(report_path, index=False)
        print(f"  - CSV Report for '{disaster}' saved: {report_path}")


def save_summary_report(fold_results, fold_detailed_info, config):
    """
    Generates complete text report of cross-validation with aggregate statistics.
    
    REPORT CONTENT:
    - Experiment configuration (model, hyperparameters, device)
    - Aggregate statistics (mean F1, std, best/worst fold)
    - Detailed results per fold (samples, F1, classification report)
    
    Args:
        fold_results: List of F1-scores for each fold
        fold_detailed_info: List of dictionaries with detailed info per fold
        config: Training configuration
        
    Returns:
        tuple: (mean_f1, std_f1)
    """
    mean_f1 = np.mean(fold_results)
    std_f1 = np.std(fold_results)
    best_f1 = np.max(fold_results)
    worst_f1 = np.min(fold_results)
    
    summary_path = os.path.join(config.results_dir, 'summary', 'cross_validation_summary.txt')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("CROSS-VALIDATION COMPLETE SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write("--- EXPERIMENT CONFIGURATION ---\n")
        f.write(f"Model:                {config.backbone_name} (pretrained={config.pretrained})\n")
        f.write(f"Number of folds (k):  {config.n_splits}\n")
        f.write(f"Epochs per fold:      {config.num_epochs}\n")
        f.write(f"Batch size:           {config.batch_size}\n")
        f.write(f"Learning rate:        {config.learning_rate}\n")
        f.write(f"Dropout rate:         {config.dropout_rate}\n")
        f.write(f"Loss function:        {config.loss_function}\n")
        f.write(f"Class weights:        {config.class_weights}\n")
        f.write(f"Device:               {config.device}\n\n")
        
        f.write("--- AGGREGATE STATISTICS (WEIGHTED F1-SCORE) ---\n")
        f.write(f"Mean F1-Score:        {mean_f1:.4f}\n")
        f.write(f"Standard Deviation:   {std_f1:.4f}\n")
        f.write(f"Best F1-Score:        {best_f1:.4f}\n")
        f.write(f"Worst F1-Score:       {worst_f1:.4f}\n\n")
        
        f.write("="*60 + "\n")
        f.write("DETAILED RESULTS PER FOLD\n")
        f.write("="*60 + "\n\n")
        
        for fold_info in fold_detailed_info:
            f.write(f"--- FOLD {fold_info['fold']} ---\n")
            f.write(f"Samples:              {fold_info['train_samples']:,} training | {fold_info['val_samples']:,} validation\n")
            f.write(f"Best F1-Score:        {fold_info['f1_score']:.4f}\n\n")
            f.write("Classification Report (on best epoch):\n")
            f.write(fold_info['classification_report'])
            f.write("\n\n")
    
    print(f"\n Complete report saved in: {summary_path}")
    return mean_f1, std_f1

def train_single_fold(fold_k, train_df, val_df, config):
    """
    Trains and validates the model for a single fold of cross-validation.
    
    COMPLETE FLOW:
    1. Setup: DataLoader, model, optimizer, loss function
    2. Training loop: Gradual unfreezing or full training
    3. Validation: Metric collection and best model saving
    4. Output: Loss curves, confusion matrices, CSV reports
    
    Args:
        fold_k: Current fold number (0 to n_splits-1)
        train_df: DataFrame with training samples
        val_df: DataFrame with validation samples
        config: Training configuration
        
    Returns:
        tuple: (best_f1, detailed_results_df, train_size, val_size, classification_report)
    """
    models_dir, plots_dir, reports_dir = setup_fold_directories(config.results_dir, fold_k)

    log_path = os.path.join(reports_dir, f'training_log_fold_{fold_k}.txt')
    original_stdout = sys.stdout
    logger = Logger(log_path)
    sys.stdout = logger  

    try:
        print(f"\n{'='*25}")
        print(f" START TRAINING FOR FOLD {fold_k} ")
        print(f"{'='*25}")

        train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(
            train_df, val_df, config)
        
        print(f"Fold {fold_k}: {len(train_dataset)} training patches, {len(val_dataset)} validation patches.")

        model = SiameseNetwork(config)
        print(f"Using device: {config.device}")
        print(f"Model: {config.backbone_name} | Classes: {config.num_classes} | Hidden: {config.hidden_size}")
        model.to(config.device)

        # Compile model for better performance 
        if hasattr(config, 'compile_model') and config.compile_model and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                print(" Model compiled with torch.compile")
            except Exception as e:
                print(f" torch.compile not available: {e}")
        
    

        if config.use_gradual_unfreezing:
            print("\n" + "="*60)
            print(" STRATEGY: GRADUAL UNFREEZING")
            print("="*60)
            print(f"Phase 1 (Epochs 1-{config.frozen_epochs}): Feature Extraction")
            print(f"  • Backbone FROZEN (fixed ImageNet weights)")
            print(f"  • Only classifier_head trained")
            print(f"  • Learning rate: {config.learning_rate}")
            print(f"\nPhase 2 (Epochs {config.frozen_epochs+1}-{config.num_epochs}): Fine-Tuning")
            print(f"  • Backbone UNFROZEN (domain adaptation)")
            print(f"  • Entire model trained")
            print(f"  • Learning rate REDUCED: {config.finetune_learning_rate}")
            print("="*60 + "\n")
            
            freeze_backbone(model)
        else:
            print("\n" + "="*60)
            print(" STRATEGY: FULL TRAINING")
            print("="*60)
            print(f"Entire model trained from start")
            print(f"Learning rate: {config.learning_rate}")
            print("="*60 + "\n")
        
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        criterion = create_loss_function(config, train_df)
        
        scaler = torch.amp.GradScaler('cuda') if config.mixed_precision and config.device.type == 'cuda' else None
        if scaler:
            print(" Mixed precision training enabled")
        
        print("Model and optimizer initialized.")
        
        best_val_f1 = 0.0
        best_detailed_df = pd.DataFrame()
        best_classification_report = ""
        train_losses = []
        val_losses = []

        for epoch in range(config.num_epochs):
            
            # TRANSITION: Feature Extraction → Fine-Tuning
            if config.use_gradual_unfreezing and epoch == config.frozen_epochs:
                print("\n" + "━"*30)
                print(" TRANSITION: Feature Extraction → Fine-Tuning")
                print("━"*30)
                
                unfreeze_backbone(model)
                
                # Recreate optimizer with reduced LR (necessary to change LR)
                optimizer = optim.AdamW(model.parameters(), lr=config.finetune_learning_rate)
                print(f" Optimizer recreated with reduced LR: {config.finetune_learning_rate}")
                
                # Recreate scaler (necessary after optimizer change)
                if scaler:
                    scaler = torch.amp.GradScaler('cuda')
                    print(" GradScaler recreated")
                
                print("━"*30 + "\n")
            
            avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config, fold_k, epoch, scaler)
            
            avg_val_loss, val_f1, all_labels, all_preds, detailed_results_df = validate_model(
                model, val_loader, criterion, config, fold_k, epoch, val_df)
            
            report = classification_report(all_labels, all_preds, target_names=config.target_names, digits=4)
            print(report)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val F1-Score: {val_f1:.4f}")

            # Save model if F1-score improves
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                model_path = os.path.join(models_dir, f'best_model_fold_{fold_k}.pth')
                
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(model.state_dict(), model_path)
                print(f" Best model saved: {model_path}")
                
                best_detailed_df = detailed_results_df.copy()
                best_classification_report = report
                report_path = os.path.join(reports_dir, f'validation_report_fold_{fold_k}.csv')
                
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                best_detailed_df.to_csv(report_path, index=False)
                print(f" Detailed report saved: {report_path}")

        # FOLD FINALIZATION
        save_loss_plot(train_losses, val_losses, plots_dir, fold_k, config)
        
        model_path = os.path.join(models_dir, f'best_model_fold_{fold_k}.pth')
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f" Best model loaded for final plots")
        else:
            print(f" Model file not found, using current state")
        
        save_confusion_matrix(model, val_loader, plots_dir, fold_k, config)

        if not best_detailed_df.empty:
            save_disaster_confusion_matrices(best_detailed_df, plots_dir, fold_k, config)
            save_disaster_reports_csv(best_detailed_df, reports_dir, fold_k)
        else:
            print("⚠ No detailed results for disaster-specific reports")

        print(f"\n{'='*60}")
        print(f" FOLD {fold_k} COMPLETED - Best F1: {best_val_f1:.4f}")
        print(f"{'='*60}\n")

    finally:
        logger.close()
        sys.stdout = original_stdout
        print(f" Log saved: {log_path}")
        
    return best_val_f1, best_detailed_df, len(train_dataset), len(val_dataset), best_classification_report

def train_model():
    """
    Main function that coordinates complete K-Fold Cross-Validation.
    
    COMPLETE PIPELINE:
    1. Load configuration (hyperparameters, device, paths)
    2. Load CSV catalog and clean data (remove 'un-classified')
    3. K-Fold Loop: train model for each fold independently
    4. Aggregate results: cross-validation statistics, disaster performance
    5. Generate final reports: summary TXT/CSV, disaster-specific analysis
    
    GENERATED OUTPUT:
    - results_dir/fold_X/: models, plots, reports per fold
    - results_dir/summary/: aggregate cross-validation and disaster performance reports
    
    Returns:
        None: Orchestration function that prints final results
    """
    config = TrainingConfig()
    
    print("START CROSS-VALIDATION")
    print(config)  # Print configuration for transparency
    
    # Show data augmentation configuration
    if config.enable_data_augmentation:
        print("\n" + "="*60)
        print(" DATA AUGMENTATION: ENABLED")
        print("="*60)
        print(f"Parameters:")
        print(f"  • Horizontal Flip: p={config.augment_horizontal_flip_prob}")
        print(f"  • Vertical Flip: p={config.augment_vertical_flip_prob}")
        print(f"  • Rotation (90°/180°/270°): p={config.augment_rotation_prob}")
        print(f"  • Color Jitter: p={config.augment_color_jitter_prob}")
        print(f"    - Brightness: ±{int(config.augment_brightness*100)}%")
        print(f"    - Contrast: ±{int(config.augment_contrast*100)}%")
        print(f"    - Saturation: ±{int(config.augment_saturation*100)}%")
        print(f"    - Hue: ±{int(config.augment_hue*100)}%")
        print("="*60)
    else:
        print("\n" + "="*60)
        print(" DATA AUGMENTATION: DISABLED")
        print("="*60)
    
    setup_directories(config.results_dir)
    
    print(f"\n Loading catalog: {config.labels_csv_path}")
    all_labels_df = pd.read_csv(config.labels_csv_path)
    print(f" Catalog loaded: {len(all_labels_df)} total patches")

    all_labels_df = all_labels_df[all_labels_df['damage'] != 'un-classified'].reset_index(drop=True)

    # K-Fold Cross-Validation Loop
    fold_results = []
    all_fold_detailed_results = []
    fold_detailed_info = []

    # Resume from specific fold (in case of interruption) 0 does the full cycle
    start_fold = 0  

    for k in range(start_fold, config.n_splits):
        val_df = all_labels_df[all_labels_df['fold'] == k]
        train_df = all_labels_df[all_labels_df['fold'] != k]
        
        best_f1, detailed_results_df, train_size, val_size, class_report = train_single_fold(k, train_df, val_df, config)
        fold_results.append(best_f1)
        
        if not detailed_results_df.empty:
            all_fold_detailed_results.append(detailed_results_df)
        
        fold_detailed_info.append({
            'fold': k,
            'f1_score': best_f1,
            'train_samples': train_size,
            'val_samples': val_size,
            'classification_report': class_report
        })

    print(f"\n{'='*60}")
    print(f" CROSS-VALIDATION COMPLETED")
    print(f"{'='*60}")
    print(f"F1-Score per fold: {fold_results}")
    
    mean_f1, std_f1 = save_summary_report(fold_results, fold_detailed_info, config)
    print(f"\n Average performance: F1-Score = {mean_f1:.4f} ± {std_f1:.4f}")


if __name__ == '__main__':
    train_model()

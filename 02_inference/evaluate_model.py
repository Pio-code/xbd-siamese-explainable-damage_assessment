"""
Script to evaluate model performance on test set,
to be run after 'inference.py' or 'inference_single_fold.py'.
Generates reports, confusion matrices and per-class metrics
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score, 
    precision_score, 
    recall_score,
    accuracy_score
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHYTON_DIR = os.path.join(SCRIPT_DIR, '..')

EXPERIMENT_NAME = 'exp_8_convenext_small'
EVALUATION_TYPE = 'ensemble'  # Options: 'ensemble' or 'single_fold'
# If EVALUATION_TYPE='single_fold', specify which fold to evaluate (0-4)
FOLD_NUMBER = 1

RESULTS_BASE_DIR = os.path.join(PHYTON_DIR, 'results')

if EVALUATION_TYPE == 'ensemble':
    EXPERIMENT_DIR = os.path.join(RESULTS_BASE_DIR, EXPERIMENT_NAME)
    PREDICTIONS_DIR = os.path.join(EXPERIMENT_DIR, 'predictions')
    PREDICTIONS_FILENAME = f'predictions_{EXPERIMENT_NAME}.csv'
    EVAL_SUFFIX = 'ensemble'
elif EVALUATION_TYPE == 'single_fold':
    EXPERIMENT_DIR = os.path.join(RESULTS_BASE_DIR, EXPERIMENT_NAME, f'fold_{FOLD_NUMBER}')
    PREDICTIONS_DIR = os.path.join(EXPERIMENT_DIR, 'predictions')
    PREDICTIONS_FILENAME = f'predictions_fold_{FOLD_NUMBER}_test.csv'
    EVAL_SUFFIX = f'fold_{FOLD_NUMBER}'
else:
    raise ValueError(f"Invalid EVALUATION_TYPE: {EVALUATION_TYPE}. Use 'ensemble' or 'single_fold'.")

PREDICTIONS_PATH = os.path.join(PREDICTIONS_DIR, PREDICTIONS_FILENAME)
EVALUATION_OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, 'evaluations')
os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)


CLASS_NAMES = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']

def plot_confusion_matrix(cm, class_names, output_path, normalize=False):
    """
    Args:
        cm: Confusion matrix
        class_names: Class names
        output_path: Output path
        normalize: If True, normalize by row (gives recall)
    """
    plt.figure(figsize=(8, 6))
    
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Confusion Matrix (Normalized)'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix'
    
    sns.heatmap(cm_display, 
                annot=True, 
                fmt=fmt, 
                cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'label': 'Percentage' if normalize else 'Count'})
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Confusion matrix saved: {output_path}")
    plt.close()



def analyze_by_disaster_type(results_df):
    """
    Analyzes performance by disaster type.
    Generates separate confusion matrices for each disaster type.
    
    Args:
        results_df: DataFrame with 'disaster_type' column already included
    """
    if 'disaster_type' not in results_df.columns:
        print("\n disaster_type not available")
        return None
    
    print("\n" + "=" * 60)
    print("ANALYSIS BY DISASTER TYPE")
    print("=" * 60)
    
    print("\n Sample distribution by disaster_type:")
    print(results_df['disaster_type'].value_counts())
    
    disaster_types = results_df['disaster_type'].dropna().unique()
    disaster_summary = []
    
    for disaster in sorted(disaster_types):
        disaster_df = results_df[results_df['disaster_type'] == disaster]
        
        print(f"\n{'─' * 60}")
        print(f" DISASTER: {disaster.upper()}")
        print(f"{'─' * 60}")
        print(f"Samples: {len(disaster_df)}")
        
        y_true = disaster_df['true_label_idx'].values
        y_pred = disaster_df['predicted_label_idx'].values
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"Accuratezza: {acc:.4f} ({acc*100:.4f}%)")
        print(f"F1-Score: {f1:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        
        disaster_summary.append({
            'disaster_type': disaster,
            'n_samples': len(disaster_df),
            'accuracy': acc,
            'f1_weighted': f1,
            'precision_weighted': prec,
            'recall_weighted': rec
        })
        
        print(f"\nClassification Report:")
        report = classification_report(y_true, y_pred, 
                                      labels=[0, 1, 2, 3],  # Specifica tutte le classi
                                      target_names=CLASS_NAMES, 
                                      digits=4, zero_division=0)
        print(report)
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]) 
        
        plt.figure(figsize=(8, 6))
        # Solo valori assoluti per disaster-specific confusion matrix
        sns.heatmap(cm, 
                    annot=True, 
                    fmt='d',  # Valori interi
                    cmap='Greens',
                    xticklabels=CLASS_NAMES, 
                    yticklabels=CLASS_NAMES,
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {disaster.upper()}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        safe_disaster_name = disaster.replace(" ", "_").lower()
        cm_path = os.path.join(EVALUATION_OUTPUT_DIR, f'confusion_matrix_{EVAL_SUFFIX}_{safe_disaster_name}.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved: {cm_path}")
        plt.close()
    
    disaster_summary_df = pd.DataFrame(disaster_summary)
    disaster_summary_path = os.path.join(EVALUATION_OUTPUT_DIR, f'disaster_summary_{EVAL_SUFFIX}.csv')
    disaster_summary_df.to_csv(disaster_summary_path, index=False)
    print(f"\nSummary saved: {disaster_summary_path}")
    
    print("\n Metrics Summary by Disaster Type:")
    print(disaster_summary_df.to_string(index=False))
    
    return results_df


def analyze_by_event_name(results_df):
    """
    Analyzes performance by individual event (e.g., hurricane-florence, tornado, etc.).
    Generates separate confusion matrices and metrics for each event.
    
    Args:
        results_df: DataFrame with 'event_name' column included
    """
    if 'event_name' not in results_df.columns:
        print("\n event_name not available")
        return None
    
    print("\n" + "=" * 60)
    print("ANALYSIS BY SPECIFIC EVENT")
    print("=" * 60)
    
    print("\n Sample distribution by event_name:")
    print(results_df['event_name'].value_counts())
    
    event_names = results_df['event_name'].dropna().unique()
    event_summary = []
    
    for event in sorted(event_names):
        event_df = results_df[results_df['event_name'] == event]
        
        if len(event_df) < 5:
            print(f"\nSkipped event '{event}': too few samples ({len(event_df)})")
            continue
        
        print(f"\n{'─' * 60}")
        print(f" EVENT: {event.upper()}")
        print(f"{'─' * 60}")
        print(f"Samples: {len(event_df)}")
        
        if 'disaster_type' in event_df.columns:
            disaster_types = event_df['disaster_type'].unique()
            print(f"Disaster type(s): {', '.join(disaster_types)}")
        
        y_true = event_df['true_label_idx'].values
        y_pred = event_df['predicted_label_idx'].values
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"F1-Score: {f1:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        
        event_summary.append({
            'event_name': event,
            'disaster_type': disaster_types[0] if 'disaster_type' in event_df.columns else 'unknown',
            'n_samples': len(event_df),
            'accuracy': acc,
            'f1_weighted': f1,
            'precision_weighted': prec,
            'recall_weighted': rec
        })
        
        print(f"\nClassification Report:")
        report = classification_report(y_true, y_pred, 
                                      labels=[0, 1, 2, 3],
                                      target_names=CLASS_NAMES, 
                                      digits=4, zero_division=0)
        print(report)
        
        # Confusion matrix only if enough samples
        if len(event_df) >= 20:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]) 
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, 
                        annot=True, 
                        fmt='d',
                        cmap='Oranges',
                        xticklabels=CLASS_NAMES, 
                        yticklabels=CLASS_NAMES,
                        cbar_kws={'label': 'Count'})
            plt.title(f'Confusion Matrix - {event.upper()}', fontsize=16, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            
            safe_event_name = event.replace(" ", "_").replace("/", "-").lower()
            cm_path = os.path.join(EVALUATION_OUTPUT_DIR, f'confusion_matrix_{EVAL_SUFFIX}_event_{safe_event_name}.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved: {cm_path}")
            plt.close()
    
    if event_summary:
        event_summary_df = pd.DataFrame(event_summary)
        event_summary_df = event_summary_df.sort_values(['disaster_type', 'accuracy'], ascending=[True, False])
        
        event_summary_path = os.path.join(EVALUATION_OUTPUT_DIR, f'event_summary_{EVAL_SUFFIX}.csv')
        event_summary_df.to_csv(event_summary_path, index=False)
        print(f"\n Event summary saved: {event_summary_path}")
        
        print("\n Metrics Summary by Event:")
        print(event_summary_df.to_string(index=False))
    
    return results_df


def main():
    """Main function."""
    print("=" * 60)
    print("TEST SET PERFORMANCE EVALUATION")
    print("=" * 60)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Evaluation type: {EVALUATION_TYPE.upper()}")
    if EVALUATION_TYPE == 'single_fold':
        print(f"Fold: {FOLD_NUMBER}")
    print("=" * 60)
    
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"\n Predictions file not found: {PREDICTIONS_PATH}")
        return
    
    print(f"\n Loading predictions...")
    results_df = pd.read_csv(PREDICTIONS_PATH)
    print(f" Samples: {len(results_df)}")
    

    y_true = results_df['true_label_idx'].values
    y_pred = results_df['predicted_label_idx'].values
    
    # GLOBAL METRICS
    print("\n" + "=" * 60)
    print("GLOBAL METRICS")
    print("=" * 60)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy:             {accuracy:.4f} ({accuracy*100:.4f}%)")  
    print(f"\nMacro-Average:")
    print(f"  Precision:             {precision_macro:.4f}")
    print(f"  Recall:                {recall_macro:.4f}")
    print(f"  F1-Score:              {f1_macro:.4f}")
    print(f"\nWeighted-Average:")
    print(f"  Precision:             {precision_weighted:.4f}")
    print(f"  Recall:                {recall_weighted:.4f}")
    print(f"  F1-Score:              {f1_weighted:.4f}")
    

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, 
                                   digits=4, zero_division=0)
    print(report)
    
    report_path = os.path.join(EVALUATION_OUTPUT_DIR, f'classification_report_{EVAL_SUFFIX}.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"TEST SET EVALUATION - {EXPERIMENT_NAME}\n")
        f.write(f"Type: {EVALUATION_TYPE.upper()}")
        if EVALUATION_TYPE == 'single_fold':
            f.write(f" - Fold {FOLD_NUMBER}")
        f.write("\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1: {f1_macro:.4f}\n")
        f.write(f"Weighted F1: {f1_weighted:.4f}\n\n")
        f.write(report)
    print(f" Report saved: {report_path}")
    
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    cm_plot_path = os.path.join(EVALUATION_OUTPUT_DIR, f'confusion_matrix_{EVAL_SUFFIX}.png')
    plot_confusion_matrix(cm, CLASS_NAMES, cm_plot_path, normalize=False)
    
    cm_plot_path_norm = os.path.join(EVALUATION_OUTPUT_DIR, f'confusion_matrix_{EVAL_SUFFIX}_normalized.png')
    plot_confusion_matrix(cm, CLASS_NAMES, cm_plot_path_norm, normalize=True)
    
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    
    errors_df = results_df[results_df['true_label_idx'] != results_df['predicted_label_idx']].copy()
    print(f" Errors: {len(errors_df)} ({len(errors_df)/len(results_df)*100:.4f}%)")
    
    errors_csv_path = os.path.join(EVALUATION_OUTPUT_DIR, f'errors_{EVAL_SUFFIX}.csv')
    errors_df.to_csv(errors_csv_path, index=False)
    print(f" Errors saved: {errors_csv_path}")
    
    if 'disaster_type' in results_df.columns:
        analyze_by_disaster_type(results_df)
    else:
        print("\n disaster_type not available in predictions CSV")
    
    if 'event_name' in results_df.columns:
        analyze_by_event_name(results_df)
    else:
        print("\n event_name not available in predictions CSV")
    
    print("\n" + "=" * 60)
    print("COMPLETED!")
    print("=" * 60)


if __name__ == '__main__':
    main()
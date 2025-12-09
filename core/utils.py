"""
Utility functions for the XBD damage classification system.

Contains:
- Logger: Logging to file and terminal
- setup_directories: Creating results directory structure
- setup_fold_directories: Creating directories for single fold
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Logger:
    """
    Class to redirect print() output to both terminal and log file.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


def setup_directories(results_dir):
    """
    Creates main 'results' folder and 'summary' subfolder
    to contain final summary of all folds.
    """
    try:
        os.makedirs(results_dir, exist_ok=True)
        summary_dir = os.path.join(results_dir, 'summary')
        os.makedirs(summary_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {e}")
        

def setup_fold_directories(results_dir, fold_k):
    """
    Creates specific folders for a single cross-validation fold.
    """
    fold_dir = os.path.join(results_dir, f'fold_{fold_k}')
    models_dir = os.path.join(fold_dir, 'models')  # Contains .pth files of best model
    plots_dir = os.path.join(fold_dir, 'plots')   
    reports_dir = os.path.join(fold_dir, 'reports')  
    
    try:
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating directories for fold {fold_k}: {e}")
    
    return models_dir, plots_dir, reports_dir


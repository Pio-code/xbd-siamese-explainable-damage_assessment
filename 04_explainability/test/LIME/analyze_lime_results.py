"""
Script for analyzing LIME results.
Reads the complete CSV and allows filtering/aggregation in various ways:
- Overall across all classes
- Only damage classes (excluding no-damage)
- By specific disaster type
- By class+disaster combinations
- Model comparisons
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Optional


class LIMEResultsAnalyzer:
    """Analyzes LIME results from CSV"""
    
    def __init__(self, results_csv_path: str):
        """
        Args:
            results_csv_path: Path to the individual_results.csv file
        """
        self.csv_path = Path(results_csv_path)
        self.output_dir = self.csv_path.parent / 'analysis'
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"  LIME RESULTS ANALYSIS")
        print(f"{'='*80}")
        print(f"  CSV: {self.csv_path}")
        print(f"  Output: {self.output_dir}")
        
        self.df = pd.read_csv(self.csv_path)
        
        print(f"\n  Loaded {len(self.df)} samples")
        print(f"  Columns: {', '.join(self.df.columns.tolist())}")
        
        required_cols = ['example_index', 'true_class', 'predicted_class', 
                        'disaster_type', 'deletion_auc', 'insertion_auc']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            print(f"\n   WARNING: Missing columns: {missing_cols}")
            print(f"    Some analyses may not work properly.")
        
        self._print_dataset_info()
        print(f"{'='*80}\n")
    
    def _print_dataset_info(self):
        """Prints basic dataset information."""
        print(f"\n  DATASET STATISTICS:")
        
        print(f"\n Class distribution (predicted):")
        class_counts = self.df['predicted_class'].value_counts()
        for cls, count in class_counts.items():
            pct = count / len(self.df) * 100
            print(f"       {cls}: {count} ({pct:.1f}%)")
        
        if 'disaster_type' in self.df.columns:
            print(f"\n     Disaster type distribution:")
            disaster_counts = self.df['disaster_type'].value_counts()
            for disaster, count in disaster_counts.items():
                pct = count / len(self.df) * 100
                print(f"       {disaster}: {count} ({pct:.1f}%)")
        
        print(f"\n Overall AUC:")
        print(f"  Deletion:  {self.df['deletion_auc'].mean():.4f} ± {self.df['deletion_auc'].std():.4f}")
        print(f"  Insertion: {self.df['insertion_auc'].mean():.4f} ± {self.df['insertion_auc'].std():.4f}")
    
    def analyze_overall(self, save_json: bool = True) -> Dict:
        """Overall analysis on all samples."""
        print(f"\n OVERALL ANALYSIS (all samples)")
        
        stats = {
            'overall': {
                'deletion_auc_mean': self.df['deletion_auc'].mean(),
                'deletion_auc_std': self.df['deletion_auc'].std(),
                'deletion_auc_median': self.df['deletion_auc'].median(),
                'insertion_auc_mean': self.df['insertion_auc'].mean(),
                'insertion_auc_std': self.df['insertion_auc'].std(),
                'insertion_auc_median': self.df['insertion_auc'].median(),
                'num_samples': len(self.df)
            }
        }
        
        print(f"   Samples: {stats['overall']['num_samples']}")
        print(f"   Deletion AUC:  {stats['overall']['deletion_auc_mean']:.4f} ± {stats['overall']['deletion_auc_std']:.4f}")
        print(f"   Insertion AUC: {stats['overall']['insertion_auc_mean']:.4f} ± {stats['overall']['insertion_auc_std']:.4f}")
        
        if save_json:
            output_path = self.output_dir / 'stats_overall.json'
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"   ✓ Saved: {output_path}")
        
        return stats
    
    def analyze_damage_only(self, save_json: bool = True) -> Dict:
        """Analysis excluding no-damage."""
        print(f"\n DAMAGE-ONLY ANALYSIS (excluding no-damage)")
        
        df_damage = self.df[self.df['predicted_class'] != 'no-damage'].copy()
        num_excluded = len(self.df) - len(df_damage)
        
        stats = {
            'damage_only': {
                'deletion_auc_mean': df_damage['deletion_auc'].mean(),
                'deletion_auc_std': df_damage['deletion_auc'].std(),
                'deletion_auc_median': df_damage['deletion_auc'].median(),
                'insertion_auc_mean': df_damage['insertion_auc'].mean(),
                'insertion_auc_std': df_damage['insertion_auc'].std(),
                'insertion_auc_median': df_damage['insertion_auc'].median(),
                'num_samples': len(df_damage),
                'num_samples_excluded': num_excluded,
                'excluded_class': 'no-damage'
            }
        }
        
        print(f"   Damage samples: {len(df_damage)}")
        print(f"   Excluded (no-damage): {num_excluded}")
        print(f"   Deletion AUC:  {stats['damage_only']['deletion_auc_mean']:.4f} ± {stats['damage_only']['deletion_auc_std']:.4f}")
        print(f"   Insertion AUC: {stats['damage_only']['insertion_auc_mean']:.4f} ± {stats['damage_only']['insertion_auc_std']:.4f}")
        
        if save_json:
            output_path = self.output_dir / 'stats_damage_only.json'
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"   ✓ Saved: {output_path}")
        
        return stats
    
    def analyze_by_disaster(self, save_json: bool = True) -> Dict:
        """Analysis by disaster type."""
        if 'disaster_type' not in self.df.columns:
            print(f"\n Column 'disaster_type' not found, skipping disaster analysis")
            return {}
        
        print(f"\n ANALYSIS BY DISASTER TYPE")
        
        stats = {'by_disaster': {}}
        
        for disaster in self.df['disaster_type'].unique():
            df_disaster = self.df[self.df['disaster_type'] == disaster]
            
            stats['by_disaster'][disaster] = {
                'deletion_auc_mean': df_disaster['deletion_auc'].mean(),
                'deletion_auc_std': df_disaster['deletion_auc'].std(),
                'insertion_auc_mean': df_disaster['insertion_auc'].mean(),
                'insertion_auc_std': df_disaster['insertion_auc'].std(),
                'num_samples': len(df_disaster)
            }
            
            print(f"\n   {disaster} (n={len(df_disaster)}):")
            print(f"     Deletion:  {stats['by_disaster'][disaster]['deletion_auc_mean']:.4f} ± {stats['by_disaster'][disaster]['deletion_auc_std']:.4f}")
            print(f"     Insertion: {stats['by_disaster'][disaster]['insertion_auc_mean']:.4f} ± {stats['by_disaster'][disaster]['insertion_auc_std']:.4f}")
        
        if save_json:
            output_path = self.output_dir / 'stats_by_disaster.json'
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\n   ✓ Saved: {output_path}")
        
        return stats
    
    def analyze_by_class(self, save_json: bool = True) -> Dict:
        """Analysis by class."""
        print(f"\n ANALYSIS BY CLASS")
        
        stats = {'by_class': {}}
        
        for class_name in self.df['predicted_class'].unique():
            df_class = self.df[self.df['predicted_class'] == class_name]
            
            stats['by_class'][class_name] = {
                'deletion_auc_mean': df_class['deletion_auc'].mean(),
                'deletion_auc_std': df_class['deletion_auc'].std(),
                'insertion_auc_mean': df_class['insertion_auc'].mean(),
                'insertion_auc_std': df_class['insertion_auc'].std(),
                'num_samples': len(df_class)
            }
            
            print(f"\n   {class_name} (n={len(df_class)}):")
            print(f"     Deletion:  {stats['by_class'][class_name]['deletion_auc_mean']:.4f} ± {stats['by_class'][class_name]['deletion_auc_std']:.4f}")
            print(f"     Insertion: {stats['by_class'][class_name]['insertion_auc_mean']:.4f} ± {stats['by_class'][class_name]['insertion_auc_std']:.4f}")
        
        if save_json:
            output_path = self.output_dir / 'stats_by_class.json'
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\n   ✓ Saved: {output_path}")
        
        return stats
    
    def analyze_by_class_and_disaster(self, save_json: bool = True) -> Dict:
        """Analysis by class + disaster type combination."""
        if 'disaster_type' not in self.df.columns:
            print(f"\n Column 'disaster_type' not found, skipping combined analysis")
            return {}
        
        print(f"\n ANALYSIS BY CLASS AND DISASTER TYPE")
        
        stats = {'by_class_and_disaster': {}}
        
        for class_name in self.df['predicted_class'].unique():
            stats['by_class_and_disaster'][class_name] = {}
            
            df_class = self.df[self.df['predicted_class'] == class_name]
            
            print(f"\n   {class_name}:")
            
            for disaster in df_class['disaster_type'].unique():
                df_subset = df_class[df_class['disaster_type'] == disaster]
                
                if len(df_subset) < 3: 
                    continue
                
                stats['by_class_and_disaster'][class_name][disaster] = {
                    'deletion_auc_mean': df_subset['deletion_auc'].mean(),
                    'deletion_auc_std': df_subset['deletion_auc'].std(),
                    'insertion_auc_mean': df_subset['insertion_auc'].mean(),
                    'insertion_auc_std': df_subset['insertion_auc'].std(),
                    'num_samples': len(df_subset)
                }
                
                print(f"     {disaster} (n={len(df_subset)}): Del={df_subset['deletion_auc'].mean():.3f}, Ins={df_subset['insertion_auc'].mean():.3f}")
        
        if save_json:
            output_path = self.output_dir / 'stats_by_class_and_disaster.json'
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\n   ✓ Saved: {output_path}")
        
        return stats
    
    def generate_visualizations(self):
        """Generates complete visualizations."""
        print(f"\n GENERATING VISUALIZATIONS")
        
        self._plot_auc_by_class()
        
        if 'disaster_type' in self.df.columns:
            self._plot_auc_by_disaster()
        
        if 'disaster_type' in self.df.columns:
            self._plot_heatmap_class_disaster()
        
        print(f"\n Visualizations completed")
    
    def _plot_auc_by_class(self):
        """Plot AUC by class."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('AUC by Class', fontsize=16, weight='bold')
        
        class_stats = self.df.groupby('predicted_class').agg({
            'deletion_auc': ['mean', 'std'],
            'insertion_auc': ['mean', 'std']
        }).reset_index()
        
        classes = class_stats['predicted_class'].tolist()
        x_pos = np.arange(len(classes))
        
        # Deletion
        del_means = class_stats['deletion_auc']['mean'].tolist()
        del_stds = class_stats['deletion_auc']['std'].tolist()
        
        axes[0].bar(x_pos, del_means, yerr=del_stds, color='red', alpha=0.7, capsize=5)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(classes, rotation=45, ha='right')
        axes[0].set_ylabel('Deletion AUC', fontsize=11, weight='bold')
        axes[0].set_title('Deletion AUC', fontsize=12, weight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Insertion
        ins_means = class_stats['insertion_auc']['mean'].tolist()
        ins_stds = class_stats['insertion_auc']['std'].tolist()
        
        axes[1].bar(x_pos, ins_means, yerr=ins_stds, color='blue', alpha=0.7, capsize=5)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(classes, rotation=45, ha='right')
        axes[1].set_ylabel('Insertion AUC', fontsize=11, weight='bold')
        axes[1].set_title('Insertion AUC', fontsize=12, weight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / 'plot_auc_by_class.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✓ Saved: {output_path}")
    
    def _plot_auc_by_disaster(self):
        """Plot AUC by disaster type."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('AUC by Disaster Type', fontsize=16, weight='bold')
        
        # Calculate means by disaster
        disaster_stats = self.df.groupby('disaster_type').agg({
            'deletion_auc': ['mean', 'std'],
            'insertion_auc': ['mean', 'std']
        }).reset_index()
        
        disasters = disaster_stats['disaster_type'].tolist()
        x_pos = np.arange(len(disasters))
        
        # Deletion
        del_means = disaster_stats['deletion_auc']['mean'].tolist()
        del_stds = disaster_stats['deletion_auc']['std'].tolist()
        
        axes[0].bar(x_pos, del_means, yerr=del_stds, color='darkred', alpha=0.7, capsize=5)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(disasters, rotation=45, ha='right')
        axes[0].set_ylabel('Deletion AUC', fontsize=11, weight='bold')
        axes[0].set_title('Deletion AUC', fontsize=12, weight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Insertion
        ins_means = disaster_stats['insertion_auc']['mean'].tolist()
        ins_stds = disaster_stats['insertion_auc']['std'].tolist()
        
        axes[1].bar(x_pos, ins_means, yerr=ins_stds, color='darkblue', alpha=0.7, capsize=5)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(disasters, rotation=45, ha='right')
        axes[1].set_ylabel('Insertion AUC', fontsize=11, weight='bold')
        axes[1].set_title('Insertion AUC', fontsize=12, weight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / 'plot_auc_by_disaster.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✓ Saved: {output_path}")
    
    def _plot_heatmap_class_disaster(self):
        """Heatmap AUC by class x disaster combination."""
        # Pivot table for deletion
        pivot_del = self.df.pivot_table(
            values='deletion_auc',
            index='predicted_class',
            columns='disaster_type',
            aggfunc='mean'
        )
        
        # Pivot table for insertion
        pivot_ins = self.df.pivot_table(
            values='insertion_auc',
            index='predicted_class',
            columns='disaster_type',
            aggfunc='mean'
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Heatmap AUC: Class x Disaster Type', fontsize=16, weight='bold')
        
        # Deletion heatmap
        sns.heatmap(pivot_del, annot=True, fmt='.3f', cmap='Reds', ax=axes[0], 
                   cbar_kws={'label': 'Deletion AUC'}, vmin=0, vmax=1)
        axes[0].set_title('Deletion AUC', fontsize=12, weight='bold')
        axes[0].set_xlabel('Disaster Type', fontsize=11, weight='bold')
        axes[0].set_ylabel('Class', fontsize=11, weight='bold')
        
        # Insertion heatmap
        sns.heatmap(pivot_ins, annot=True, fmt='.3f', cmap='Blues', ax=axes[1],
                   cbar_kws={'label': 'Insertion AUC'}, vmin=0, vmax=1)
        axes[1].set_title('Insertion AUC', fontsize=12, weight='bold')
        axes[1].set_xlabel('Disaster Type', fontsize=11, weight='bold')
        axes[1].set_ylabel('Class', fontsize=11, weight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / 'heatmap_class_disaster.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✓ Saved: {output_path}")
    
    def run_all_analyses(self):
        """Runs all available analyses."""
        print(f"\n RUNNING COMPLETE ANALYSES\n")
        
        # Base analyses
        self.analyze_overall()
        self.analyze_damage_only()
        self.analyze_by_class()
        
        # Disaster analyses (if available)
        if 'disaster_type' in self.df.columns:
            self.analyze_by_disaster()
            self.analyze_by_class_and_disaster()
        
        # Visualizations
        self.generate_visualizations()
        
        print(f"\n{'='*80}")
        print(f"✓ ANALYSES COMPLETED!")
        print(f"{'='*80}")
        print(f"  Outputs saved in: {self.output_dir}")
        print(f"{'='*80}\n")


def main():
    """Main function with argparse."""
    parser = argparse.ArgumentParser(
        description='Analyze LIME results with maximum flexibility'
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to the individual_results.csv file'
    )
    
    parser.add_argument(
        '--analysis',
        type=str,
        default='all',
        choices=['all', 'overall', 'damage_only', 'by_disaster', 'by_class', 
                'by_class_and_disaster', 'visualizations'],
        help='Type of analysis to run'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = LIMEResultsAnalyzer(args.csv)
    
    # Run requested analysis
    if args.analysis == 'all':
        analyzer.run_all_analyses()
    elif args.analysis == 'overall':
        analyzer.analyze_overall()
    elif args.analysis == 'damage_only':
        analyzer.analyze_damage_only()
    elif args.analysis == 'by_disaster':
        analyzer.analyze_by_disaster()
    elif args.analysis == 'by_class':
        analyzer.analyze_by_class()
    elif args.analysis == 'by_class_and_disaster':
        analyzer.analyze_by_class_and_disaster()
    elif args.analysis == 'visualizations':
        analyzer.generate_visualizations()


if __name__ == '__main__':
    main()

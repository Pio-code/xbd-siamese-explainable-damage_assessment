"""
Visualization of xBD dataset distribution for exploratory analysis.

Generates:
1. Bar charts for each disaster type (damage class distribution)
2. General pie chart (building percentage per class)
3. Heatmap of damage distribution per disaster

Output: Saves PNG figures in the 'data_prep/visualizations/' folder
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHYTON_DIR = os.path.join(SCRIPT_DIR, '..', '..')
LABELS_CSV_PATH = os.path.join(PHYTON_DIR, 'patched_dataset_unified', 'labels.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'visualizations')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DAMAGE_COLORS = {
    'no-damage': '#2ecc71',
    'minor-damage': '#f39c12',
    'major-damage': '#e74c3c',
    'destroyed': '#8e44ad'
}
DAMAGE_ORDER = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']

DISASTER_COLORS = {
    'fire': '#e74c3c',
    'flooding': '#3498db',
    'wind': '#95a5a6',
    'earthquake': '#8b4513',
    'tsunami': '#1abc9c',
    'volcano': '#d35400'
}


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    print(f" Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'damage' in df.columns:
        initial_count = len(df)
        df = df[df['damage'] != 'un-classified'].copy()
        removed = initial_count - len(df)
        if removed > 0:
            print(f"   Removed {removed:,} unclassified buildings")
    
    print(f"  Dataset loaded: {len(df):,} buildings (patches)")
    print(f"   Disaster types: {df['disaster_type'].nunique()}")
    if 'fold' in df.columns:
        print(f"   Folds: {df['fold'].nunique()}")
    
    return df


def calculate_building_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates building statistics by disaster type from labels.csv file.
    Returns a DataFrame with columns: disaster_type, damage_class, count
    """
    stats_list = []
    
    for disaster in df['disaster_type'].unique():
        df_disaster = df[df['disaster_type'] == disaster]
        
        damage_counts = df_disaster['damage'].value_counts()
        
        for damage_class in DAMAGE_ORDER:
            count = damage_counts.get(damage_class, 0)
            stats_list.append({
                'disaster_type': disaster,
                'damage_class': damage_class,
                'count': count
            })
    
    return pd.DataFrame(stats_list)


def plot_pie_chart_overall(df: pd.DataFrame, output_path: str):
    """
    Pie chart with building distribution by disaster type.
    """
    print("\n Creating pie chart")
    
    disaster_buildings = df['disaster_type'].value_counts().to_dict()
    
    disaster_buildings = dict(sorted(disaster_buildings.items(), key=lambda x: x[1], reverse=True))
    
    total_buildings = sum(disaster_buildings.values())
    percentages = {k: (v / total_buildings) * 100 for k, v in disaster_buildings.items()}
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = [DISASTER_COLORS.get(disaster.lower(), '#95a5a6') for disaster in disaster_buildings.keys()]
    
    wedges, texts, autotexts = ax.pie(
        disaster_buildings.values(), 
        labels=None,  
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 16, 'weight': 'bold'},  
        pctdistance=0.85
    )
    
    legend_labels = [f"{disaster.title()}: {count:,} buildings ({percentages[disaster]:.1f}%)" 
                    for disaster, count in disaster_buildings.items()]
    
    ax.legend(wedges, legend_labels,
              title="Disaster Type",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=15,
              title_fontsize=16)

    ax.set_title('Building Distribution by Disaster Type\n(Complete xBD Dataset)',
                 fontsize=18, weight='bold', pad=20)

    textstr = f'Total\n{total_buildings:,}\nbuildings'
    ax.text(0, 0, textstr, transform=ax.transData,
            fontsize=17, ha='center', va='center', weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=2))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_path}")
    plt.close()


def plot_bars_per_disaster(df: pd.DataFrame, output_dir: str):
    print("\n Creating grouped bar chart by disaster type...")
    
    building_stats = calculate_building_stats(df)
    
    pivot_df = building_stats.pivot(index='disaster_type', columns='damage_class', values='count').fillna(0)
    pivot_df = pivot_df.reindex(columns=DAMAGE_ORDER)
    
    pivot_df['total'] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values('total', ascending=False)
    pivot_df = pivot_df.drop('total', axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    disasters = pivot_df.index
    x = np.arange(len(disasters))
    width = 0.2
    
    # Create VERTICAL bars for each damage class
    for idx, damage_class in enumerate(DAMAGE_ORDER):
        offset = (idx - 1.5) * width
        values = pivot_df[damage_class].values
        bars = ax.bar(x + offset, values, width, 
                      label=damage_class.replace('-', ' ').title(),
                      color=DAMAGE_COLORS[damage_class],
                      alpha=0.85,
                      edgecolor='black',
                      linewidth=0.5)
        
        # Add values above all bars 
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width()/2, val + 1000,
                   f'{int(val):,}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold',
                   rotation=90)  
    
    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in disasters], fontsize=12, weight='bold', rotation=0)
    ax.set_ylabel('Number of Buildings', fontsize=13, fontweight='bold')
    ax.set_title('Damage Class Distribution by Disaster Type', 
                 fontsize=16, fontweight='bold', pad=15)
    
    ax.legend(title='Damage Class', fontsize=11, title_fontsize=12,
              loc='upper right', framealpha=0.95, edgecolor='black')
    
    ax.grid(axis='y', alpha=0.5, linestyle='-', linewidth=0.7)
    ax.set_axisbelow(True)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{int(y/1000):.0f}k'))
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'damage_distribution_per_disaster.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_path}")
    plt.close()


def plot_heatmap_disaster_damage(df: pd.DataFrame, output_path: str):

    print("\n Creating damage distribution heatmap...")
    
    building_stats = calculate_building_stats(df)
    
    pivot_df = building_stats.pivot(
        index='disaster_type', 
        columns='damage_class', 
        values='count'
    )
    
    pivot_df = pivot_df[DAMAGE_ORDER]
    
    pivot_pct = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
    
    pivot_pct = pivot_pct.sort_values('destroyed', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(
        pivot_pct, 
        annot=True, 
        fmt='.1f',
        cmap='RdYlGn_r',  
        cbar_kws={'label': 'Percentage (%)'},
        linewidths=1,
        linecolor='gray',
        ax=ax
    )
    
    ax.set_title('Percentage Damage Distribution by Disaster Type\n(% buildings per class)', 
                 fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('Damage Class', fontsize=12, weight='bold')
    ax.set_ylabel('Disaster Type', fontsize=12, weight='bold')
    
    ax.set_xticklabels([col.replace('-', ' ').title() for col in pivot_pct.columns], rotation=45, ha='right')
    ax.set_yticklabels([disaster.title() for disaster in pivot_pct.index], rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_path}")
    plt.close()


def main():
    """Main function to generate all visualizations."""
    
    print("="*80)
    print(" xBD DATASET VISUALIZATION GENERATION")
    print("="*80)
    
    df = load_and_prepare_data(LABELS_CSV_PATH)
    
    print(" Generating charts...")
    
    plot_pie_chart_overall(
        df, 
        os.path.join(OUTPUT_DIR, 'overall_damage_distribution_pie.png')
    )
    
    plot_bars_per_disaster(df, OUTPUT_DIR)
    
    plot_heatmap_disaster_damage(
        df,
        os.path.join(OUTPUT_DIR, 'heatmap_damage_per_disaster.png')
    )
    
    print("\n" + "="*80)
    print(" PROCESS COMPLETED!")
    print(f" Charts saved in: {OUTPUT_DIR}")
    print("="*80)
    
    print("\n Generated files:")
    for filename in sorted(os.listdir(OUTPUT_DIR)):
        if filename.endswith('.png'):
            print(f"  • {filename}")


if __name__ == '__main__':
    main()

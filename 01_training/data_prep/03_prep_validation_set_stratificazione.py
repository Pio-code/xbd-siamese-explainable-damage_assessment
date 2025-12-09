"""
Script to add folds for Stratified Group K-Fold Cross-Validation.
Stratify to maintain the distribution of damage and disaster type.
Group to avoid leakage between train and validation sets, therefore:
"""

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

xbd_enriched_df: pd.DataFrame = pd.read_csv('xbd_train_val_catalog_enriched.csv')
print("Dataset loaded.")
print(f"Total number of scenes: {len(xbd_enriched_df)}")
print(f"Disaster types in the dataset: {xbd_enriched_df['disaster_type'].value_counts()}")


def create_damage_profile_category_v3(scene_row: pd.Series) -> str:
    """
    Classifies a scene into a damage profile category based on building counts.

    scene_row:
        A DataFrame row representing a single scene. Expected columns:
        - 'no-damage', 'minor-damage', 'major-damage', 'destroyed'
    """

    total_buildings = scene_row['no-damage'] + scene_row['minor-damage'] + scene_row['major-damage'] + scene_row['destroyed']
    
    if total_buildings == 0:
        return 'no_buildings'
    
    destroyed_pct = scene_row['destroyed'] / total_buildings
    major_damage_pct = scene_row['major-damage'] / total_buildings
    minor_damage_pct = scene_row['minor-damage'] / total_buildings
    no_damage_pct = scene_row['no-damage'] / total_buildings
    
    # Aggregates
    severe_damage_pct = major_damage_pct + destroyed_pct
    any_damage_pct = 1 - no_damage_pct


    if destroyed_pct >= 0.25 or severe_damage_pct >= 0.5:
        return 'extreme_destruction'
        
    elif destroyed_pct >= 0.05 or severe_damage_pct >= 0.15:
        return 'severe_widespread_damage'
    
    elif any_damage_pct >= 0.5:
        return 'moderate_damage'
        
    elif any_damage_pct >= 0.1:
        return 'light_widespread_damage'
        
    elif any_damage_pct > 0:
        return 'sporadic_damage'
        
    else:
        return 'undamaged'

def create_stratification_target(scene_row: pd.Series) -> str:
    """
    Creates a composite target for stratification that combines:
    1. Disaster type
    2. Damage profile
    """
    disaster_type = scene_row['disaster_type']
    damage_profile = scene_row['damage_profile']
    
    return f"{disaster_type}_{damage_profile}"


print("\n CREATING CATEGORIES FOR STRATIFICATION ")

xbd_enriched_df['damage_profile'] = xbd_enriched_df.apply(create_damage_profile_category_v3, axis=1)

print("Damage profile distribution:")
print(xbd_enriched_df['damage_profile'].value_counts())

xbd_enriched_df['stratify_target'] = xbd_enriched_df.apply(create_stratification_target, axis=1)

print(f"\nNumber of unique composite categories: {xbd_enriched_df['stratify_target'].nunique()}")

# Handle categories with few instances
min_samples_per_category = 5  
category_counts = xbd_enriched_df['stratify_target'].value_counts()
rare_categories = category_counts[category_counts < min_samples_per_category].index

if len(rare_categories) > 0:
    print(f"\n {len(rare_categories)} rare categories grouped.")
    
    def handle_rare_categories(scene_row):
        if scene_row['stratify_target'] in rare_categories:
            return f"{scene_row['disaster_type']}_rare"
        return scene_row['stratify_target']
    
    xbd_enriched_df['stratify_target'] = xbd_enriched_df.apply(handle_rare_categories, axis=1)
    
    print(f"Number of final categories: {xbd_enriched_df['stratify_target'].nunique()}")




N_SPLITS: int = 5

print(f"\n STRATIFIED GROUP K-FOLD CROSS-VALIDATION (k={N_SPLITS})")

# Input for the splitting function
X: pd.DataFrame = xbd_enriched_df 
y: pd.Series = xbd_enriched_df['stratify_target']  
groups: pd.Series = xbd_enriched_df['scene_id']  
print(f"Number of unique groups (scenes): {xbd_enriched_df['scene_id'].nunique()}")
print(f"Number of categories for stratification: {y.nunique()}")


sgkf= StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# New column to store the fold number for each scene
xbd_enriched_df['fold'] = -1

# Perform the split and assign the fold number to each row
try:
    for fold_num, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        xbd_enriched_df.loc[val_idx, 'fold'] = fold_num
        print(f"Fold {fold_num}: {len(val_idx)} scenes assigned")
    
    print("Split completed")
    
except Exception as e:
    print(f"ERROR during split: {e}")

print("\n  SPLIT QUALITY VERIFICATION")

print("Number of scenes per fold:")
fold_counts = xbd_enriched_df['fold'].value_counts().sort_index()
print(fold_counts)

print("\nPercentage of each disaster type per fold:")
disaster_by_fold = pd.crosstab(xbd_enriched_df['fold'], xbd_enriched_df['disaster_type'])
disaster_percentages = pd.crosstab(xbd_enriched_df['fold'], xbd_enriched_df['disaster_type'], normalize='index') * 100
print(disaster_percentages.round(1))


print("\n SPLIT QUALITY METRICS")
damage_by_fold = xbd_enriched_df.groupby('fold')[['no-damage', 'minor-damage', 'major-damage', 'destroyed']].sum()
damage_percentages = damage_by_fold.div(damage_by_fold.sum(axis=1), axis=0) * 100
damage_profile_by_fold = pd.crosstab(xbd_enriched_df['fold'], xbd_enriched_df['damage_profile'])

disaster_var = disaster_percentages.var().mean()
damage_var = damage_percentages.var().mean()

print(f"Mean variance of disaster proportions across folds: {disaster_var:.2f}")
print(f"Mean variance of damage proportions across folds: {damage_var:.2f}")
print("(Lower values indicate better stratification)")


if xbd_enriched_df['fold'].min() == -1:
    print("\nWARNING: Some scenes were not assigned to a fold. Check the data.")
else:
    xbd_enriched_df.to_csv("xbd_catalog_with_folds.csv", index=False)

    print(f"\nColumns in final dataset:")
    print(xbd_enriched_df.columns.tolist())
    
    report_path = "stratification_quality_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("XBD DATASET STRATIFICATION QUALITY REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generation date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total number of scenes: {len(xbd_enriched_df)}\n")
        f.write(f"Number of folds: {N_SPLITS}\n")
        f.write(f"Strategy: Stratified Group K-Fold Cross-Validation\n\n")
        
        f.write("DATASET COMPOSITION\n")
        f.write("-" * 40 + "\n")
        f.write("Disaster type distribution:\n")
        disaster_counts = xbd_enriched_df['disaster_type'].value_counts()
        for disaster, count in disaster_counts.items():
            pct = (count / len(xbd_enriched_df)) * 100
            f.write(f"  {disaster}: {count} scenes ({pct:.1f}%)\n")
        
        f.write(f"\nDamage profile distribution:\n")
        damage_profile_counts = xbd_enriched_df['damage_profile'].value_counts()
        for profile, count in damage_profile_counts.items():
            pct = (count / len(xbd_enriched_df)) * 100
            f.write(f"  {profile}: {count} scenes ({pct:.1f}%)\n")
        
        f.write(f"\nTotal building statistics per damage class:\n")
        total_building_stats = xbd_enriched_df[['no-damage', 'minor-damage', 'major-damage', 'destroyed', 'un-classified']].sum()
        total_buildings = total_building_stats.sum()
        f.write(f"  Total buildings in dataset: {total_buildings:,}\n")
        for damage_class, count in total_building_stats.items():
            pct = (count / total_buildings) * 100
            f.write(f"  {damage_class}: {count:,} buildings ({pct:.1f}%)\n")
        
        f.write(f"\nNumber of composite categories for stratification: {xbd_enriched_df['stratify_target'].nunique()}\n")
        
        if len(rare_categories) > 0:
            f.write(f"\nRare categories grouped ({len(rare_categories)} categories):\n")
            for cat in rare_categories:
                f.write(f"  - {cat}: {category_counts[cat]} scenes\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("DISTRIBUTION PER FOLD\n")
        f.write("=" * 80 + "\n")
        
        f.write("Number of scenes per fold:\n")
        for fold, count in fold_counts.items():
            f.write(f"  Fold {fold}: {count} scenes\n")
        
        f.write(f"\nDisaster type distribution per fold:\n")
        f.write(disaster_by_fold.to_string())
        f.write(f"\n\nDisaster type percentages per fold:\n")
        f.write(disaster_percentages.round(1).to_string())
        
        f.write(f"\n\nBuilding distribution per damage class per fold:\n")
        f.write(damage_by_fold.to_string())
        f.write(f"\n\nBuilding percentages per damage class per fold:\n")
        f.write(damage_percentages.round(1).to_string())
        
        f.write(f"\n\nDamage profile distribution per fold:\n")
        f.write(damage_profile_by_fold.to_string())
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("QUALITY METRICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Mean variance of disaster proportions across folds: {disaster_var:.4f}\n")
        f.write(f"Mean variance of damage proportions across folds: {damage_var:.4f}\n")
        
        fold_size_var = fold_counts.var()
        f.write(f"Fold size variance: {fold_size_var:.2f}\n")
        f.write(f"Fold size coefficient of variation: {fold_counts.std()/fold_counts.mean()*100:.2f}%\n\n")
        

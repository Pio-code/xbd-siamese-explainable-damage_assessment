"""
Script to enrich xbd_test_catalog.csv with disaster_type and damage counts
equivalent to 02_prep_validation_set_conteggio_danni.py from training
"""

import os
import pandas as pd
import json
from typing import Dict, Any
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_CSV_PATH = os.path.join(SCRIPT_DIR, 'xbd_test_catalog.csv')
TEST_LABELS_DIR = os.path.join(SCRIPT_DIR, '..', '..', '..', 'test', 'labels')


def get_damage_counts_and_disaster_type(post_json_path: str) -> Dict[str, Any]:
    """
    Opens a post_disaster JSON file and counts the number of buildings for each damage class.
    Also extracts the disaster type from metadata.
    """
    result: Dict[str, Any] = {
        'no-damage': 0, 
        'minor-damage': 0, 
        'major-damage': 0, 
        'destroyed': 0, 
        'un-classified': 0, 
        'disaster_type': None
    }
    
    if not os.path.exists(post_json_path):
        print(f" JSON not found: {post_json_path}")
        return result
    
    try:
        with open(post_json_path, 'r') as f:
            data = json.load(f)

        if 'metadata' in data and 'disaster_type' in data['metadata']:
            result['disaster_type'] = data['metadata']['disaster_type']
        
        if 'features' in data and 'xy' in data['features']:
            buildings = data['features']['xy']
            for building in buildings:
                if 'properties' in building and 'subtype' in building['properties']:
                    damage_type = building['properties']['subtype']
                    if damage_type in result:
                        result[damage_type] += 1
        
    except Exception as e:
        print(f" Error reading {post_json_path}: {e}")
    
    return result


def main():
    print("=" * 80)
    print("ENRICHING XBD_TEST_CATALOG.CSV - COMPLETE VERSION")
    print("=" * 80)
    
    if not os.path.exists(CATALOG_CSV_PATH):
        print(f"\n ERROR: File not found: {CATALOG_CSV_PATH}")
        return
    
    if not os.path.exists(TEST_LABELS_DIR):
        print(f"\n ERROR: JSON directory not found: {TEST_LABELS_DIR}")
        return
    
    print(f"\n Loading: {CATALOG_CSV_PATH}")
    xbd_test_df = pd.read_csv(CATALOG_CSV_PATH)
    print(f" Scenes loaded: {len(xbd_test_df)}")
    
    tqdm.pandas(desc="Analyzing JSON files for damage counts and disaster_type")
    
    damage_counts_and_disaster_list = xbd_test_df['post_json_path'].progress_apply(
        get_damage_counts_and_disaster_type
    )
    
    print("\n Creating DataFrame with extracted data...")
    damage_counts_df = pd.DataFrame(damage_counts_and_disaster_list.tolist())
    df_enriched = pd.concat([xbd_test_df, damage_counts_df], axis=1)
    
    unknown_count = (df_enriched['disaster_type'].isna() | (df_enriched['disaster_type'] == 'None')).sum()
    if unknown_count > 0:
        print(f"\n Warning: {unknown_count} scenes without disaster_type")
    
    print("\n" + "=" * 80)
    print("DISASTER_TYPE DISTRIBUTION:")
    print("=" * 80)
    disaster_counts = df_enriched['disaster_type'].value_counts()
    print(disaster_counts.to_string())
    
    print("\n" + "=" * 80)
    print("DAMAGE COUNT STATISTICS PER SCENE:")
    print("=" * 80)
    damage_cols = ['no-damage', 'minor-damage', 'major-damage', 'destroyed', 'un-classified']
    damage_stats = df_enriched[damage_cols].describe()
    print(damage_stats.to_string())
    
    output_path = CATALOG_CSV_PATH.replace('.csv', '_enriched.csv')
    df_enriched.to_csv(output_path, index=False)
    print(f"\n Enriched file saved: {output_path}")
    
    print("\n" + "=" * 80)
    print("FINAL STRUCTURE (columns):")
    print("=" * 80)
    print(df_enriched.columns.tolist())
    
    print("\n" + "=" * 80)
    print("FIRST 3 SCENES FROM ENRICHED CATALOG:")
    print("=" * 80)
    cols_to_show = ['scene_id', 'disaster_type', 'no-damage', 'minor-damage', 'major-damage', 'destroyed']
    print(df_enriched[cols_to_show].head(3).to_string())


if __name__ == '__main__':
    main()

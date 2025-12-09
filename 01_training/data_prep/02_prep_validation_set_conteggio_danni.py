"""
Script to enrich the original CSV file with damage counts and
disaster type from the JSON file, to prepare data
for Stratified Group K-Fold Cross-Validation
"""

import pandas as pd
import json
from typing import Any, Dict
from tqdm import tqdm  


xbd_enriched_df: pd.DataFrame = pd.read_csv('xbd_train_val_catalog.csv')


def get_damage_counts_and_disaster_type(json_path: str) -> Dict[str, Any]:
    """Opens a JSON file and counts the number of buildings for each damage class.
    Also extracts the disaster type from metadata.
    Returns a dictionary with counts and disaster_type."""

    result: Dict[str, Any] = {'no-damage': 0, 'minor-damage': 0, 'major-damage': 0, 'destroyed': 0, 'un-classified': 0, 'disaster_type': None}
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        if 'metadata' in data and 'disaster_type' in data['metadata']:
            result['disaster_type'] = data['metadata']['disaster_type']

        buildings = data['features']['xy']  
        for building in buildings:
            damage_type = building['properties']['subtype']
            if damage_type in result:
                result[damage_type] += 1
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
    return result



tqdm.pandas(desc="Analyzing JSON files for damage counts and disaster_type")

# progress_apply is the progress bar version of apply.
# apply takes a function and applies it to each element of the Series (the 'post_json_path' column).
damage_counts_and_disaster_list: pd.Series = xbd_enriched_df['post_json_path'].progress_apply(get_damage_counts_and_disaster_type)


damage_counts_df = pd.DataFrame(damage_counts_and_disaster_list.tolist())
df_enriched = pd.concat([xbd_enriched_df, damage_counts_df], axis=1)

print(df_enriched.head())


df_enriched.to_csv('xbd_train_val_catalog_enriched.csv', index=False)

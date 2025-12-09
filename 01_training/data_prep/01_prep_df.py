"""
Script to create a Pandas DataFrame from the xBD dataset
with complete paths for each scene (images and labels)
"""

import os 
import pandas as pd

def create_dataset_catalog(base_dir):
    """
    Scans the xBD data structure with separate 'images' and 'labels' directories
    to create a DataFrame that associates each scene with its file paths.
    """
    scenes_data = []

    train_dir = os.path.join(base_dir, 'train')

    for tier in ['tier1', 'tier3']:
        tier_path = os.path.join(train_dir, tier)
        
        images_dir = os.path.join(tier_path, 'images')
        labels_dir = os.path.join(tier_path, 'labels')
        
        if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
            print(f"'images' or 'labels' directories not found in {tier_path}.")
            continue

        # Iterate through files in the images directory to extract scene prefixes
        for filename in os.listdir(images_dir):
            # Use only one file type (e.g., _post_disaster) to avoid processing each scene 4 times
            if filename.endswith('_post_disaster.png'):

                scene_prefix = filename.replace('_post_disaster.png', '')
                
                post_image_path = os.path.join(images_dir, filename) 
                pre_image_path = os.path.join(images_dir, f"{scene_prefix}_pre_disaster.png")
                post_json_path = os.path.join(labels_dir, f"{scene_prefix}_post_disaster.json")
                pre_json_path = os.path.join(labels_dir, f"{scene_prefix}_pre_disaster.json")
                
                if all([
                    os.path.exists(post_image_path), 
                    os.path.exists(pre_image_path), 
                    os.path.exists(post_json_path), 
                    os.path.exists(pre_json_path)
                ]):
                    scenes_data.append({
                        'scene_id': scene_prefix, 
                        'pre_image_path': pre_image_path,
                        'post_image_path': post_image_path,
                        'pre_json_path': pre_json_path,
                        'post_json_path': post_json_path
                    })

    df = pd.DataFrame(scenes_data)
    # For safety, there should be no duplicates
    df = df.drop_duplicates(subset='scene_id').reset_index(drop=True)
    return df

# Execution

try:
    dataset_catalog_df = create_dataset_catalog("xbd")

    if not dataset_catalog_df.empty:
        print("Catalog created successfully")
        print(f"Found {len(dataset_catalog_df)} complete scenes.")
        print("\nHere are the first 5 rows of the catalog:")
        print(dataset_catalog_df.head())
        
        output_csv_path = "xbd_train_val_catalog.csv"
        dataset_catalog_df.to_csv(output_csv_path, index=False)
        print(f"\nCatalog successfully saved to file: {output_csv_path}")
        
    else:
        print("No scenes found.")

except FileNotFoundError:
    print("\nERROR: The specified directory was not found.")

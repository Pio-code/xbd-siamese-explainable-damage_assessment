"""
Creating the dataframe for the TEST dataset
This script is analogous to data_preprocessing.py, but operates on the test folder
"""

import os 
import pandas as pd

def create_test_catalog(base_dir):
    """
    Scans the xBD data structure in the 'test' folder
    to create a DataFrame that associates each scene with its file paths.
    """
    scenes_data = []

    test_dir = os.path.join(base_dir, 'test')
    
    if not os.path.isdir(test_dir):
        print(f"ERROR: Test directory not found in {test_dir}")
        return pd.DataFrame()
    
    images_dir = os.path.join(test_dir, 'images')
    labels_dir = os.path.join(test_dir, 'labels')
    
    if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
        scenes_data.extend(process_directory(images_dir, labels_dir))
    else:
        print(f"ERROR: 'images' or 'labels' directories not found in {test_dir}")
        return pd.DataFrame()


    df = pd.DataFrame(scenes_data)
    df = df.drop_duplicates(subset='scene_id').reset_index(drop=True)
    return df


def process_directory(images_dir, labels_dir):
    """
    Processes a pair of images/labels directories and returns a list of scenes.
    """
    scenes_data = []
    
    for filename in os.listdir(images_dir):
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
    
    return scenes_data



if __name__ == '__main__':
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        phyton_dir = os.path.join(script_dir, '..', '..')
        base_dir = os.path.join(phyton_dir, '..')
        
        print("=" * 60)
        print("CREATING TEST SET CATALOG")
        print("=" * 60)
        
        test_catalog_df = create_test_catalog(base_dir)

        if not test_catalog_df.empty:
            print("\n Test catalog created successfully")
            print(f" Found {len(test_catalog_df)} complete scenes in the test set.")
            print("\nFirst 5 scenes from test catalog:")
            print(test_catalog_df.head())
            
            output_csv_path = os.path.join(script_dir, 'xbd_test_catalog.csv')
            test_catalog_df.to_csv(output_csv_path, index=False)
            print(f"\n Catalog saved successfully: {output_csv_path}")
            print(f"  (Saved in: 02_inference/data_prep_test/)")
            
        else:
            print("\n ERROR: No scenes found. Check the directory structure.")

    except FileNotFoundError as e:
        print(f"\n ERROR: The specified directory was not found: {e}")
    except Exception as e:
        print(f"\n UNEXPECTED ERROR: {e}")

"""
Creating patches for the TEST SET
This script is analogous to patching.py but operates on test data
"""

import os
import json
import cv2
import pandas as pd
from shapely.wkt import loads as wkt_loads
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_CATALOG_PATH = os.path.join(SCRIPT_DIR, 'xbd_test_catalog_enriched.csv')
BASE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'patched_dataset_test')
PATCH_SIZE = 128


def parse_wkt(wkt_string):
    '''
    Converts a WKT string into a Shapely Polygon object.
    Returns None if conversion fails.
    '''
    try:
        return wkt_loads(wkt_string)
    except Exception:
        return None


def crop_patch(image, polygon, patch_size):
    '''
    Crops a square patch centered on the building polygon.
    Adds black padding if necessary to ensure patch_size x patch_size dimensions.
    '''
    
    min_x, min_y, max_x, max_y = polygon.bounds
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    half_size = patch_size / 2
    
    y1 = int(center_y - half_size)
    y2 = int(center_y + half_size)
    x1 = int(center_x - half_size)
    x2 = int(center_x + half_size)
    
    patch = image[max(0, y1):min(image.shape[0], y2),
                   max(0, x1):min(image.shape[1], x2)]
    
    pad_y1 = max(0, -y1)
    pad_y2 = max(0, y2 - image.shape[0])
    pad_x1 = max(0, -x1)
    pad_x2 = max(0, x2 - image.shape[1])
    
    if any([pad_y1, pad_y2, pad_x1, pad_x2]):
        patch = cv2.copyMakeBorder(patch, pad_y1, pad_y2, pad_x1, pad_x2, 
                                   cv2.BORDER_CONSTANT, value=0)
    
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

    return patch



def main():
    print("=" * 60)
    print("CREATING PATCHES FOR TEST SET")
    print("=" * 60)
    
    if not os.path.exists(BASE_CATALOG_PATH):
        print(f"\n ERROR: Test catalog not found: {BASE_CATALOG_PATH}")
        print("Run 'data_preprocessing_test.py' first to create the catalog.")
        return
    
    df_catalog = pd.read_csv(BASE_CATALOG_PATH)
    print(f"\n Catalog loaded: {len(df_catalog)} test scenes")

    output_img_dir = os.path.join(BASE_OUTPUT_DIR, 'images')
    os.makedirs(output_img_dir, exist_ok=True)
    print(f" Output directory created: {output_img_dir}")

    output_labels = []
    total_patches = 0
    skipped_unclassified = 0
    errors = 0

    print("\nStarting patch extraction...")
    
    for _, scene_row in tqdm(df_catalog.iterrows(), total=len(df_catalog), desc="Processing scenes"):
        try:
            img_pre = cv2.imread(scene_row['pre_image_path'])
            img_post = cv2.imread(scene_row['post_image_path'])
            
            if img_pre is None or img_post is None:
                print(f"\n Error loading images for scene {scene_row['scene_id']}")
                errors += 1
                continue
            
            with open(scene_row['post_json_path'], 'r') as f:
                labels_data = json.load(f)
            
            buildings = labels_data['features']['xy']
            
            for building in buildings: 
                props = building['properties']
                damage_label = props['subtype']
                building_uid = props['uid']
                
                if damage_label == 'un-classified':
                    skipped_unclassified += 1
                    continue
                
                polygon = parse_wkt(building['wkt'])
                if polygon is None or polygon.is_empty:
                    continue

                patch_pre = crop_patch(img_pre, polygon, PATCH_SIZE)
                patch_post = crop_patch(img_post, polygon, PATCH_SIZE)

                patch_id = f"{scene_row['scene_id']}_{building_uid}"
                pre_patch_filename = f"{patch_id}_pre.png"
                post_patch_filename = f"{patch_id}_post.png"
                
                pre_patch_path = os.path.join(output_img_dir, pre_patch_filename)
                post_patch_path = os.path.join(output_img_dir, post_patch_filename)
                
                cv2.imwrite(pre_patch_path, patch_pre)
                cv2.imwrite(post_patch_path, patch_post)

                output_labels.append({
                    'pre_patch_path': os.path.join('images', pre_patch_filename),
                    'post_patch_path': os.path.join('images', post_patch_filename),
                    'damage': damage_label,
                    'disaster_type': scene_row['disaster_type']
                })
                
                total_patches += 1

        except Exception as e:
            print(f"\n Error processing scene {scene_row['scene_id']}: {e}")
            errors += 1
    
    labels_df = pd.DataFrame(output_labels)
    labels_csv_path = os.path.join(BASE_OUTPUT_DIR, 'labels_test.csv')
    labels_df.to_csv(labels_csv_path, index=False)
    
    print("\n" + "=" * 60)
    print("TEST PATCH CREATION SUMMARY")
    print("=" * 60)
    print(f" Total patches created: {total_patches}")
    print(f" 'un-classified' buildings skipped: {skipped_unclassified}")
    print(f" Errors during processing: {errors}")
    print(f" Labels file created: {labels_csv_path}")
    print("\nDamage class distribution in test set:")
    print(labels_df['damage'].value_counts())
    print("=" * 60)


if __name__ == '__main__':
    main()

""" 
Subdivision of scenes into patches where each scene will be divided into a number
of images equivalent to the number of buildings.
each patch will be centered on the building and will be 128x128 pixels
each patch will inherit the damage label of the central building
"""


import os
import pandas as pd
import json
import cv2
from shapely.wkt import loads as wkt_loads
from tqdm import tqdm

BASE_CATALOG_PATH = 'xbd_catalog_with_folds.csv'
BASE_OUTPUT_DIR = 'patched_dataset_unified' 
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
    Crops a square patch of size patch_size x patch_size centered on the building.
    If the patch extends beyond the image borders, adds black padding to complete it.
    
    Logic in two phases:
    1. CROP: Crops only the part that exists in the image (may be smaller than patch_size)
    2. PAD: If the crop is incomplete, adds black borders to reach patch_size x patch_size
    '''

    half_size = patch_size / 2
    
    min_x, min_y, max_x, max_y = polygon.bounds
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    y1 = int(center_y - half_size)
    y2 = int(center_y + half_size)
    x1 = int(center_x - half_size)
    x2 = int(center_x + half_size)
    
    # PHASE 1: CROP
    # max(0, y1) -> if y1 is negative (outside top border), start from 0
    # min(h, y2) -> if y2 exceeds height, stop at bottom border
    patch = image[max(0, y1):min(image.shape[0], y2), 
                  max(0, x1):min(image.shape[1], x2)]
    
    # PHASE 2: PAD - Calculate how much padding is needed to complete the patch
    # If y1 was -10, 10 pixels are missing at the top -> pad_y1 = 10
    pad_y1 = max(0, -y1)
    # If y2 was 1030 and image.height is 1024, 6 pixels are missing at the bottom -> pad_y2 = 6
    pad_y2 = max(0, y2 - image.shape[0])
    # Same reasoning for left and right
    pad_x1 = max(0, -x1)
    pad_x2 = max(0, x2 - image.shape[1])

  
    if any([pad_y1, pad_y2, pad_x1, pad_x2]):
        patch = cv2.copyMakeBorder(patch, pad_y1, pad_y2, pad_x1, pad_x2, cv2.BORDER_CONSTANT, value=0)
        
    # Due to possible "off-by-one" rounding errors, sometimes the patch
    # may be 1 pixel smaller or larger than expected (as happened to me).
    # cv2.resize ensures that the output is always the exact size.
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

    return patch


def main():
    print("Starting unified patching process...")
    df_catalog = pd.read_csv(BASE_CATALOG_PATH)

    output_img_dir = os.path.join(BASE_OUTPUT_DIR, 'images')
    os.makedirs(output_img_dir, exist_ok=True)

    output_labels = []

    print(f"Creating patches for all {len(df_catalog)} scenes...")

    for _, scene_row in tqdm(df_catalog.iterrows(), total=len(df_catalog)):
        try:
            # Load pre and post disaster images into memory using OpenCV.
            img_pre = cv2.imread(scene_row['pre_image_path'])
            img_post = cv2.imread(scene_row['post_image_path'])
            
            with open(scene_row['post_json_path'], 'r') as f:
                labels_data = json.load(f)
            
            buildings = labels_data['features']['xy']
            
            for building in buildings:
                props = building['properties']
                damage_label = props['subtype']
                building_uid = props['uid']
                
                if damage_label == 'un-classified':
                    continue
                
                polygon = parse_wkt(building['wkt'])
                if polygon is None or polygon.is_empty:
                    continue

                patch_pre = crop_patch(img_pre, polygon, PATCH_SIZE)
                patch_post = crop_patch(img_post, polygon, PATCH_SIZE)
                
                base_filename = f"{scene_row['scene_id']}_{building_uid}"
                pre_patch_filename = f"{base_filename}_pre.png"
                post_patch_filename = f"{base_filename}_post.png"
                
     
                cv2.imwrite(os.path.join(output_img_dir, pre_patch_filename), patch_pre)
                cv2.imwrite(os.path.join(output_img_dir, post_patch_filename), patch_post)
                
                # This dictionary contains all the necessary information for
                # model training.
                output_labels.append({
                    'pre_patch_path': os.path.join('images', pre_patch_filename),
                    'post_patch_path': os.path.join('images', post_patch_filename),
                    'damage': damage_label,
                    'disaster_type': scene_row['disaster_type'],
                    'fold': scene_row['fold'] 
                })

        except Exception as e:
            print(f"Error processing scene {scene_row['scene_id']}: {e}")
    
    labels_df = pd.DataFrame(output_labels)
    labels_df.to_csv(os.path.join(BASE_OUTPUT_DIR, 'labels.csv'), index=False)
    print(f"\nCreated unified 'labels.csv' file with {len(labels_df)} patches.")
    print("\nPatching process completed.")

if __name__ == '__main__':
    main()
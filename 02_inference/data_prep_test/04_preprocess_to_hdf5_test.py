"""
Conversion of test dataset to HDF5 format
This script is analogous to preprocess_to_hdf5.py but operates on test data
"""
import os
import h5py
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(SCRIPT_DIR, 'patched_dataset_test')
LABELS_CSV_PATH = os.path.join(BASE_DATA_DIR, 'labels_test.csv')

OUTPUT_HDF5_PATH = os.path.join(BASE_DATA_DIR, 'xbd_test_dataset.hdf5')

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3


def create_hdf5_test_dataset():
    """
    Reads the test dataset from PNG files and saves it in a single HDF5 file
    """
    print("=" * 60)
    print("CONVERTING TEST SET TO HDF5")
    print("=" * 60)
    
    if not os.path.exists(LABELS_CSV_PATH):
        print(f"\n ERROR: Labels file not found: {LABELS_CSV_PATH}")
        print("Run 'patching_test.py' first to create test patches.")
        return
    
    print(f"\n Loading catalog from: {LABELS_CSV_PATH}")
    df = pd.read_csv(LABELS_CSV_PATH)
    
    print(f"Number of samples before cleanup: {len(df)}")
    df = df[df['damage'] != 'un-classified'].reset_index(drop=True)
    num_samples = len(df)
    print(f"Number of samples after cleanup: {num_samples}")
    
    if num_samples == 0:
        print("\n ERROR: No valid samples found in test catalog!")
        return
    
    with h5py.File(OUTPUT_HDF5_PATH, 'w') as hf:
        print(f"\n Creating HDF5 file: {OUTPUT_HDF5_PATH}")

        dset_pre = hf.create_dataset('pre_images', 
                                     (num_samples, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                                     dtype=np.uint8,
                                     chunks=(64, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)) 
                                     
        dset_post = hf.create_dataset('post_images', 
                                      (num_samples, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                                      dtype=np.uint8,
                                      chunks=(64, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
                                      
        dset_labels = hf.create_dataset('labels', 
                                        (num_samples,), 
                                        dtype=np.int64)
        
        class_to_idx = {'no-damage': 0, 'minor-damage': 1, 'major-damage': 2, 'destroyed': 3}

        print("\nStarting HDF5 conversion...")
        invalid_samples = 0
        
        for idx, row in tqdm(df.iterrows(), total=num_samples, desc="Converting"):
            pre_img_path = os.path.join(BASE_DATA_DIR, row['pre_patch_path'])
            post_img_path = os.path.join(BASE_DATA_DIR, row['post_patch_path'])

            try:
                with Image.open(pre_img_path).convert('RGB') as img_pre:
                    dset_pre[idx] = np.array(img_pre, dtype=np.uint8)

                with Image.open(post_img_path).convert('RGB') as img_post:
                    dset_post[idx] = np.array(img_post, dtype=np.uint8)

                label_str = row['damage']
                dset_labels[idx] = class_to_idx[label_str]

            except Exception as e:
                print(f"\n WARNING: Error processing index {idx} ({pre_img_path}): {e}")
                dset_pre[idx] = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
                dset_post[idx] = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
                dset_labels[idx] = -1  
                invalid_samples += 1
    
    print("\n" + "=" * 60)
    print("HDF5 CONVERSION SUMMARY")
    print("=" * 60)
    print(f" Process completed!")
    print(f" Valid samples: {num_samples - invalid_samples}")
    print(f" Samples with errors: {invalid_samples}")
    print(f" Dataset saved: {OUTPUT_HDF5_PATH}")
    print("=" * 60)

if __name__ == '__main__':
    create_hdf5_test_dataset()

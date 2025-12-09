""" 
Converts the patched image dataset into a single HDF5 file
for faster access during model training.
the HDF5 file saves data in Numpy arrays in a binary format
"""

import os
from typing import Dict
import h5py
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm


SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
PHYTON_DIR = os.path.join(SCRIPT_DIR, '..')
BASE_DATA_DIR: str = os.path.join(PHYTON_DIR, 'patched_dataset_unified')
LABELS_CSV_PATH: str = os.path.join(BASE_DATA_DIR, 'labels.csv')

OUTPUT_HDF5_PATH: str = os.path.join(BASE_DATA_DIR, 'xbd_dataset.hdf5')

IMG_HEIGHT: int = 128
IMG_WIDTH: int = 128
IMG_CHANNELS: int = 3


def create_hdf5_dataset() -> None:
    """
    Reads the dataset from PNG files and saves it into a single HDF5 file
    Removes samples with 'un-classified' label before conversion.
    """
    print(f"Loading catalog from: {LABELS_CSV_PATH}")
    df = pd.read_csv(LABELS_CSV_PATH)

    print(f"Number of samples before cleaning: {len(df)}")
    df = df[df['damage'] != 'un-classified'].reset_index(drop=True)  
    num_samples = len(df)
    print(f"Number of samples after cleaning: {num_samples}")
    
    with h5py.File(OUTPUT_HDF5_PATH, 'w') as hf:
        print(f"Creating HDF5 file at: {OUTPUT_HDF5_PATH}")

        # (num_samples, ...) is the pre-allocation. Instead of creating an empty dataset
        # and adding data bit by bit, it prepares disk space upfront
        # large enough to contain num_samples images of size 128x128x3"
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
        
        class_to_idx: Dict[str, int] = {'no-damage': 0, 'minor-damage': 1, 'major-damage': 2, 'destroyed': 3}

        print("\nStarting conversion process...")
        for idx, sample_row in tqdm(df.iterrows(), total=num_samples, desc="Converting to HDF5"):
            pre_img_path: str = os.path.join(BASE_DATA_DIR, sample_row['pre_patch_path'])
            post_img_path: str = os.path.join(BASE_DATA_DIR, sample_row['post_patch_path'])

            try:
                with Image.open(pre_img_path).convert('RGB') as img_pre:  # .convert('RGB') for safety
                    # the image is converted to a NumPy array and inserted into slot idx
                    # of the 'pre_images' dataset
                    dset_pre[idx] = np.array(img_pre, dtype=np.uint8)

                with Image.open(post_img_path).convert('RGB') as img_post:
                    dset_post[idx] = np.array(img_post, dtype=np.uint8)

                label_str: str = sample_row['damage']
                dset_labels[idx] = class_to_idx[label_str]

            except Exception as e:
                print(f"\nWARNING: Error processing index {idx} ({pre_img_path}). Will be skipped. Error: {e}")
                # these samples will be filled with zeros
                dset_pre[idx] = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
                dset_post[idx] = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
                dset_labels[idx] = -1  # Invalid label
                
    print("\nProcess completed!")
    print(f"Dataset successfully saved to '{OUTPUT_HDF5_PATH}'.")

if __name__ == '__main__':
    create_hdf5_dataset()
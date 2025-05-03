import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
from pathlib import Path

JSON_PATH = Path('/home/yd344/dvornek_10t/Datasets/SA-Med2D/raw/MeCoVQA/SAMed2Dv1/SAMed2D_v1.json')
LOCAL_IMAGE_DIR = Path('/home/yd344/dvornek_10t/Datasets/SA-Med2D/raw/MeCoVQA/SAMed2Dv1/images')
OUTPUT_DIR = Path('/home/yd344/dvornek_10t/Datasets/SA-Med2D/raw/MeCoVQA/SAMed2Dv1/masks_merged')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def merge_masks(mask_list):
    """
    Merges a list of mask images into a single mask image.

    Args:
        mask_list (list): List of file paths to the mask images.

    Returns:
        np.ndarray: Merged mask image.
    """
    base_mask_name = ''.join(p.split('---')[:-1])
    mask_cls2idx_dict = {}
    for p in mask_list:
        cls, idx = p.replace('.png', '').split('---')[-1].split('_')
        if cls not in mask_cls2idx_dict:
            mask_cls2idx_dict[cls] = []
        mask_cls2idx_dict[cls].append(idx)
    
    # remove keys with only one value
    mask_cls2idx_dict = {k: v for k, v in mask_cls2idx_dict.items() if len(v) > 1}
    
    
    for cls in mask_cls2idx_dict.keys():
        merged_mask = None
        for idx in mask_cls2idx_dict[cls]:
            mask_path = os.path.join('masks', f"{base_mask_name}---{cls}_{idx}.png")
            if not os.path.exists(mask_path):
                print(f"Mask file {mask_path} does not exist.")
                continue

            # Open the mask image
            mask_image = Image.open(mask_path).convert("L")
            mask_array = np.array(mask_image)

            if merged_mask is None:
                merged_mask = np.zeros_like(mask_array)
            
            merged_mask = np.bitwise_or(merged_mask, mask_array)
        output_path = os.path.join(OUTPUT_DIR, f"{base_mask_name}---{cls}_merged.png")
        Image.fromarray(merged_mask).save(output_path)


if __name__ == "__main__":
    # Example usage
    image_list = glob(os.path.join(LOCAL_IMAGE_DIR, '*.png'))
    image2mask = json.load(open(JSON_PATH, 'r'))
    
    for image_path in tqdm(image_list):
        mask_list = image2mask.get(image_path.replace('images/', ''), [])
        merge_masks(mask_list)
    
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
from pathlib import Path
import multiprocessing as mp

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
    
    # Split masks according to their class
    base_mask_name = ''.join(mask_list[0].split('---')[:-1])
    mask_cls2idx_dict = {}
    for p in mask_list:
        cls, idx = p.replace('.png', '').split('---')[-1].split('_')
        if cls not in mask_cls2idx_dict:
            mask_cls2idx_dict[cls] = []
        mask_cls2idx_dict[cls].append(idx)
    
    # remove keys with only one value
    mask_cls2idx_dict = {k: v for k, v in mask_cls2idx_dict.items() if len(v) > 1}

    # Merge masks from the same class
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
            if mask_image.max() > 1:
                mask_array = mask_array // 255
            mask_array = mask_array.astype(np.uint8)

            if merged_mask is None:
                merged_mask = np.zeros_like(mask_array)
            
            merged_mask = np.bitwise_or(merged_mask, mask_array)
        if merged_mask is None:
            print(f"No valid masks found for {base_mask_name}---{cls}.")
            continue
        output_path = os.path.join(OUTPUT_DIR, f"{base_mask_name}---{cls}_merged.png")
        if merged_mask.max() <= 1.0:
            merged_mask = merged_mask * 255
        merged_mask = merged_mask.astype(np.uint8)
        Image.fromarray(merged_mask).save(output_path)


def process_image_chunk(chunk_data):
    """
    Process a chunk of images
    
    Args:
        chunk_data (tuple): Tuple containing (chunk of image paths, image2mask dictionary)
    """
    image_chunk, image2mask = chunk_data
    for image_path in tqdm(image_chunk, total=len(image_chunk), desc="Processing images"):
        print(image_path)
        mask_list = image2mask.get(image_path, [])
        if len(mask_list) == 0:
            continue
        merge_masks(mask_list)
    return len(image_chunk)

def split_list(lst, n):
    """Split list into n roughly equal chunks"""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

if __name__ == "__main__":
    # Load data
    image_list = glob(os.path.join(LOCAL_IMAGE_DIR, '*.png'))
    image_list = [img.replace('/home/yd344/dvornek_10t/Datasets/SA-Med2D/raw/MeCoVQA/SAMed2Dv1/', '') for img in image_list]
    image2mask = json.load(open(JSON_PATH, 'r'))
    
    # Determine number of processes and chunk size
    num_processes = 2
    
    # Split images into chunks
    image_chunks = split_list(image_list, num_processes)
    chunk_args = [(chunk, image2mask) for chunk in image_chunks]
    
    print(f"Processing {len(image_list)} images using {num_processes} processes in {len(image_chunks)} chunks")
    
    # Create multiprocessing pool and process chunks
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_image_chunk, chunk_args)
    
    print(f"Completed processing {sum(results)} images")

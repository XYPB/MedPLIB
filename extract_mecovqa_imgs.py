import os
import shutil
import zipfile
import bisect
import json
from tqdm import tqdm
from pathlib import Path
from glob import glob

ZIP_PATH   = Path('/home/yd344/dvornek_10t/Datasets/SA-Med2D/raw/{full}.zip')     # the giant file on disk
OUT_DIR    = Path("/home/yd344/dvornek_10t/Datasets/SA-Med2D/raw/MeCoVQA")       # where the images will land
CHUNK      = 4 << 20                      # 1 MiB per read – tune as you like
JSON_PATH = Path('/home/yd344/dvornek_10t/Datasets/SA-Med2D/raw/MeCoVQA/SAMed2Dv1/SAMed2D_v1.json')
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".gif",
              ".bmp", ".tif", ".tiff", ".webp"}

def get_all_required_images():
    target_json = glob("./data/MeCoVQA/*/*.json")
    all_images = set(['SAMed2Dv1/SAMed2D_v1_class_mapping_id.json', 'SAMed2Dv1/SAMed2D_v1.json'])
    for json_file in target_json:
        with open(json_file, 'r') as f:
            data = json.load(f)
            for item in data:
                image_path = item['image']
                if image_path.startswith('images/'):
                    all_images.add('SAMed2Dv1/' + image_path)
    return sorted(list(all_images))


def get_all_required_masks(image_list):
    assert os.path.exists(JSON_PATH)
    image2mask = json.load(open(JSON_PATH, 'r'))
    all_masks = set()
    for image_path in image_list:
        image_path = image_path.replace('SAMed2Dv1/', '')
        masks = image2mask.get(image_path, [])
        for mask in masks:
            all_masks.add('SAMed2Dv1/' + mask)
    return sorted(list(all_masks))


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    required_images = get_all_required_images()
    required_masks = get_all_required_masks(required_images)
    print(f"Total number of required images: {len(required_images)}")
    print(f"Total number of required masks: {len(required_masks)}")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        # get directory tree
        zf_list = zf.namelist()
        print(f"Total number of files in zip: {len(zf_list)}")
        # get all required masks
        # extract the image
        for idx, image_path in tqdm(enumerate(required_images + required_masks)):
            try:
                # Extract the image from the zip file
                zf.extract(image_path, OUT_DIR)
                # print(f"Extracted: {image_path}")
            except KeyError:
                print(f"KeyError: {image_path} not found in the zip file.")
            except Exception as e:
                print(f"Error extracting {image_path}: {e}")
            if idx > 10:
                break
    print(f"All required images have been extracted to {OUT_DIR}")


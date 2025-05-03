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
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".gif",
              ".bmp", ".tif", ".tiff", ".webp"}

def upper_bound(prefix):
    # any string that is *just after* all strings beginning with prefix
    # trick: append a Unicode code‑point larger than any real character
    return prefix[:-1] + chr(ord(prefix[-1]) + 1) if prefix else chr(0x10FFFF)

def find_with_prefix(names_sorted, prefix):
    lo = bisect.bisect_left(names_sorted, prefix)
    hi = bisect.bisect_left(names_sorted, upper_bound(prefix))
    return names_sorted[lo:hi]

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

def get_mask_query(image_list):
    return [p.replace('images/', 'masks/').replace('.png', '') for p in image_list]

def get_all_required_masks(zf_list, image_list):
    query_prefix = get_mask_query(image_list)
    zf_list_sorted = sorted(zf_list)
    required_masks = find_with_prefix(zf_list_sorted, query_prefix)
    print(f"Total number of required masks: {len(required_masks)}")
    print(f"Required masks: {required_masks[:10]}")
    return required_masks


def merge_masks():
    pass

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    required_images = get_all_required_images()
    print(f"Total number of required images: {len(required_images)}")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        # get directory tree
        zf_list = zf.namelist()
        print(f"Total number of files in zip: {len(zf_list)}")
        # get all required masks
        required_masks = get_all_required_masks(zf_list, required_images[:2])
        # extract the image
        for image_path in tqdm(required_images):
            try:
                # Extract the image from the zip file
                zf.extract(image_path, OUT_DIR)
                # print(f"Extracted: {image_path}")
            except KeyError:
                print(f"KeyError: {image_path} not found in the zip file.")
            except Exception as e:
                print(f"Error extracting {image_path}: {e}")
            break
    print(f"All required images have been extracted to {OUT_DIR}")


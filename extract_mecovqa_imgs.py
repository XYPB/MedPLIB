import shutil
import zipfile
import json
from tqdm import tqdm
from pathlib import Path
from glob import glob

ZIP_PATH   = Path('/home/yd344/dvornek_10t/Datasets/SA-Med2D/raw/{full}.zip')     # the giant file on disk
OUT_DIR    = Path("/home/yd344/dvornek_10t/Datasets/SA-Med2D/raw/MeCoVQA")       # where the images will land
CHUNK      = 4 << 20                      # 1 MiB per read – tune as you like
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".gif",
              ".bmp", ".tif", ".tiff", ".webp"}

def get_all_required_images():
    target_json = glob("./data/MeCoVQA/*/*.json")
    all_images = set(['SAMed2D_v1_class_mapping_id.json', 'SAMed2D_v1.json'])
    for json_file in target_json:
        with open(json_file, 'r') as f:
            data = json.load(f)
            for item in data:
                image_path = item['image']
                all_images.add(image_path)
                # conversation = item['conversations']
                # for conv in conversation:
                #     ans_value = conv['value']
                #     if '<mask>' in ans_value:
                #         # Extract the image path from the answer value
                #         start_index = ans_value.index('<mask>') + len('<mask>')
                #         end_index = ans_value.index('</mask>')
                #         image_path = ans_value[start_index:end_index]
                #         all_images.add(image_path)
    return sorted(list(all_images))


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
        print(f"First 10 files in zip: {zf_list[:10]}")
        # extract the image
        for image_path in tqdm(required_images):
            try:
                # Extract the image from the zip file
                zf.extract(image_path, OUT_DIR)
                print(f"Extracted: {image_path}")
            except KeyError:
                print(f"KeyError: {image_path} not found in the zip file.")
            except Exception as e:
                print(f"Error extracting {image_path}: {e}")
            break
    print(f"All required images have been extracted to {OUT_DIR}.")


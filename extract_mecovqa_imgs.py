import os
import zipfile
import json
from tqdm import tqdm
from pathlib import Path
from glob import glob
import multiprocessing as mp

ZIP_PATH   = Path('/home/yd344/dvornek_10t/Datasets/SA-Med2D/raw/{full}.zip')     # the giant file on disk
OUT_DIR    = Path("/home/yd344/dvornek_10t/Datasets/SA-Med2D/raw/MeCoVQA")       # where the images will land
CHUNK      = 4 << 20                      # 1 MiB per read – tune as you like
JSON_PATH = Path('/home/yd344/dvornek_10t/Datasets/SA-Med2D/raw/MeCoVQA/SAMed2Dv1/SAMed2D_v1.json')
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".gif",
              ".bmp", ".tif", ".tiff", ".webp"}

def get_all_required_images():
    target_json = glob("./data/MeCoVQA/*/*.json")
    print(target_json)
    all_images = set(['SAMed2Dv1/SAMed2D_v1_class_mapping_id.json', 'SAMed2Dv1/SAMed2D_v1.json'])
    cnt = 0
    for json_file in target_json:
        with open(json_file, 'r') as f:
            print(f"Loading {json_file}")
            data = json.load(f)
            for item in data:
                image_path = item['image']
                if image_path.startswith('images/'):
                    all_images.add('SAMed2Dv1/' + image_path)
            print(f"Total number of images in {json_file}: {len(all_images)-cnt}")
            cnt = len(all_images)
    print(f"Total number of images in JSON files: {len(all_images)}")
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

def split_list(lst, n):
    """Split list into n roughly equal chunks"""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def extract_images_chunk(args):
    """Worker function to extract a chunk of images using its own zipfile handle"""
    chunk_images, process_id = args
    print(f"Process {process_id} started with {len(chunk_images)} images.")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for image_path in tqdm(chunk_images, total=len(chunk_images)):
            try:
                zf.extract(image_path, OUT_DIR)
            except KeyError:
                print(f"KeyError: {image_path} not found in the zip file.")
            except Exception as e:
                print(f"Error extracting {image_path}: {e}")

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # find missing images
    image_list = json.load(open("data/MeCoVQA/train/MeCoVQA-Complex_local+Region_fixed_filtered_missing_images.json", 'r'))
    image_list = ["SAMed2Dv1/" + img for img in image_list]
    image_list = sorted(image_list)
    image_list = [img + '.png' if 'png' not in img else img for img in image_list]  # Ensure all images have .png extension
    print(image_list[:10])
    print(len(image_list))
    
    extract_images_chunk((image_list, 0))  # Test the first chunk extraction

    # required_images = get_all_required_images()
    # required_masks = get_all_required_masks(required_images)
    # print(f"Total number of required images: {len(required_images)}")
    # print(f"Total number of required masks: {len(required_masks)}")
    # total_images = sorted(required_images + required_masks)
    # print(f"Total number of images to extract: {len(total_images)}")
    # # total_images = total_images[:64]
    
    # total_images_todo = []
    # for img in total_images:
    #     full_path = os.path.join(OUT_DIR, img)
    #     if not os.path.exists(full_path):
    #         total_images_todo.append(img)
    # print(f"Actual total number of images remains to be extracted: {len(total_images_todo)}")
    # total_images = total_images_todo
    
    # exit()
    
    # # Determine number of CPU cores to use
    # num_cores = 1
    # print(f"Using {num_cores} CPU cores for parallel extraction")
    
    # # Split work across cores
    # image_chunks = split_list(total_images, num_cores)
    # chunk_sum = sum([len(c) for c in image_chunks])
    # assert chunk_sum == len(total_images), f"Chunk sum {chunk_sum} does not match total images {len(total_images)}"
    
    # # Create process arguments with IDs
    # process_args = [(image_chunks[i], i) for i in range(num_cores)]
    
    # extract_images_chunk(process_args[0])  # Test the first chunk extraction
    # # Use Pool to manage processes
    # # with mp.Pool(processes=num_cores) as pool:
    #     # pool.map(extract_images_chunk, process_args)
        
    # print(f"All required images have been extracted to {OUT_DIR}")


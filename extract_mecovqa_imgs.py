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

def prefix_sweep(names, prefixes):
    """
    names      - iterable of strings
    prefixes   - iterable of strings
    Return dict{prefix → list_of_names_starting_with_it}
    """
    names_sorted  = sorted(names)
    pref_sorted   = sorted([(p, i) for i, p in enumerate(prefixes)])
    result        = [[] for _ in prefixes]

    n_idx = 0
    for p, orig_idx in pref_sorted:
        # advance n_idx to first name ≥ p
        while n_idx < len(names_sorted) and names_sorted[n_idx] < p:
            n_idx += 1

        # collect while name starts with p
        j = n_idx
        m = names_sorted
        while j < len(m) and m[j].startswith(p):
            result[orig_idx].append(m[j])
            j += 1

    return {p: result[i] for i, p in enumerate(prefixes)}

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
    return [p.replace('images/', 'masks/').replace('.png', '') for p in image_list  if 'json' not in p]

def get_all_required_masks(zf_list, image_list):
    query_prefix = get_mask_query(image_list)
    zf_list_sorted = sorted(zf_list)
    required_masks = prefix_sweep(zf_list_sorted, query_prefix)
    print(f"Total number of required masks: {len(required_masks)}")
    print(f"Required masks: {required_masks[query_prefix[0]]}")
    return required_masks


def get_all_required_masks_parallel(zf_list, image_list):
    # use get_all_required_masks function to get the required masks in a multi-process way
    from multiprocessing import Pool
    from functools import partial
    N = 4
    # chunk the image_list into N chunks
    chunk_size = len(image_list) // N
    chunks = [image_list[i:i + chunk_size] for i in range(0, len(image_list), chunk_size)]
    with Pool(N) as pool:
        # Use partial to pass the zf_list and image_list to the function
        func = partial(get_all_required_masks, zf_list)
        results = pool.map(func, chunks)
        # combine the results
        required_masks = {}
        for result in results:
            for k, v in result.items():
                if k in required_masks:
                    required_masks[k].extend(v)
                else:
                    required_masks[k] = v
    print(f"Total number of required masks: {len(required_masks)}")
    print(f"Required masks: {required_masks}")
    return required_masks


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
        required_masks = get_all_required_masks_parallel(zf_list, required_images[:16])
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


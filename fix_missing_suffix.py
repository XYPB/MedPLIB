import os
import json
from tqdm import tqdm
from glob import glob

target_json = glob("./data/MeCoVQA/*/*.json")
IMG_FOLDER = "/home/yd344/dvornek_10t/Datasets/SA-Med2D/raw/MeCoVQA/SAMed2Dv1/"

missing_images = set()
for json_file in target_json:
    with open(json_file, 'r') as f:
        print(f"Loading {json_file}")
        data = json.load(f)
        fixed_data = []
        fixed_cnt = 0
        for item in data:
            image_path = item['image']
            if image_path.startswith('images/'):
                if not image_path.endswith('.png'):
                    fixed_path = image_path + '.png'
                    fixed_item = item
                    fixed_item['image'] = fixed_path
                    fixed_data.append(fixed_item)
                    fixed_cnt += 1
                    if not os.path.exists(os.path.join(IMG_FOLDER, fixed_path)):
                        missing_images.add(image_path + '.png')
                    continue
            fixed_data.append(item)
        assert len(fixed_data) == len(data), "Data length mismatch after fixing image paths."
        if fixed_cnt > 0:
            print(f"Total number of missing images in {json_file}: {fixed_cnt}")
            output_json = json_file.replace('.json', '_fixed.json')
            with open(output_json, 'w') as f:
                json.dump(fixed_data, f, indent=2)

print(f"Total number of missing images: {len(missing_images)}")
# Save the missing images to a new JSON file
if len(missing_images) > 0:
    output_json = "./data/MeCoVQA/missing_images.json"
    with open(output_json.replace('.json', '_missing_images.json'), 'w') as f:
        json.dump(list(missing_images), f)
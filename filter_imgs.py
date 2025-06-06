import os
import json
from tqdm import tqdm

target_json = "/home/yd344/dvornek_10t/Datasets/MeCoVQA/train/MeCoVQA-Complex_local+Region_fixed.json"
data = json.load(open(target_json, 'r'))

IMG_FOLDER = "/home/yd344/dvornek_10t/Datasets/SA-Med2D/raw/MeCoVQA/SAMed2Dv1/"

filtered_data = []
missing_images = set()
cnt = 0
for item in tqdm(data):
    image_path = item['image']
    local_path = os.path.join(IMG_FOLDER, image_path)
    if not os.path.exists(local_path):
        missing_images.add(image_path)
        continue
    if image_path.startswith('images/'):
        cnt += 1

    for conversation in item['conversations']:
        value = conversation['value']
        if "<region>" in str(value):
            mask_path = value.split("<region>")[1].split("</region>")[0]
            local_mask_path = os.path.join(IMG_FOLDER, mask_path)
            if not os.path.exists(local_mask_path):
                missing_images.add(mask_path)
                continue
    filtered_data.append(item)

print(cnt)
print(f"Total number of data: {len(data)}")
print(f"Total number of data after filtering: {len(filtered_data)}")
print(f"Total number of missing images: {len(missing_images)}")
print(f"Missing images: {list(missing_images)[:10]}")
# Save the filtered data to a new JSON file
if len(missing_images) > 0:
    output_json = target_json.replace('.json', '_filtered.json')
    with open(output_json, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    with open(output_json.replace('.json', '_missing_images.json'), 'w') as f:
        json.dump(list(missing_images), f)
import os
import json
from tqdm import tqdm

target_json = "/home/yd344/dvornek_10t/Datasets/MeCoVQA/train/MeCoVQA-Complex_local+Region.json"
data = json.load(open(target_json, 'r'))

filtered_data = []
missing_images = set()
cnt = 0
for item in tqdm(data):
    image_path = item['image']
    if not os.path.exists(image_path):
        missing_images.add(image_path)
        continue
    if image_path.startswith('images/'):
        cnt += 1
    filtered_data.append(item)

print(cnt)
print(f"Total number of images: {len(data)}")
print(f"Total number of images after filtering: {len(filtered_data)}")
print(f"Total number of missing images: {len(missing_images)}")
print(f"Missing images: {missing_images[:100]}")
# Save the filtered data to a new JSON file
if len(missing_images) > 0:
    output_json = target_json.replace('.json', '_filtered.json')
    with open(output_json, 'w') as f:
        json.dump(filtered_data, f, indent=2)
import os
import json
from tqdm import tqdm

target_json = "/home/yd344/dvornek_10t/Datasets/MeCoVQA/train/MeCoVQA-Complex_local+Region.json"
data = json.load(open(target_json, 'r'))

filtered_data = []
missing_images = []
for item in tqdm(data):
    image_path = item['image']
    if not os.path.exists(image_path):
        missing_images.append(image_path)
        continue
    filtered_data.append(item)
    
print(f"Total number of images: {len(data)}")
print(f"Total number of images after filtering: {len(filtered_data)}")
print(f"Total number of missing images: {len(missing_images)}")
# Save the filtered data to a new JSON file
output_json = "/home/yd344/dvornek_10t/Datasets/MeCoVQA/train/MeCoVQA-Complex_local+Region_filtered.json"
with open(output_json, 'w') as f:
    json.dump(filtered_data, f, indent=2)
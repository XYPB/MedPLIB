import torch
import json
import os
from PIL import Image
from tqdm import tqdm
from transformers import pipeline

from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration



if __name__ == "__main__":
    # dummy test

    model_id = "google/medgemma-4b-it"

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # Image attribution: Stillwaterising, CC0, via Wikimedia Commons
    image_path = "/home/yd344/dvornek_10t/Datasets/SA-Med2D/raw/MeCoVQA/SAMed2Dv1/images/mr_t2w--BrainTumour--BRATS_106--z_0138.png"
    image = Image.open(image_path)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Can you provide an description of the overall findings from this image?"},
                {"type": "image", "image": image},
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)

    target_output = "The sagittal MRI scan reveals the presence of edema and non-enhancing and enhancing tumors in the head and neck region. Edema is characterized by abnormal accumulation of fluid, while tumors show abnormal growth in the tissues. Further analysis and treatment planning are recommended based on these findings."


import torch
import json
import os
import datetime
from PIL import Image
from tqdm import tqdm
from transformers import pipeline
from copy import deepcopy

from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModel
from transformers import Qwen2_5_VLForConditionalGeneration

torch.set_float32_matmul_precision('high')
torch.backends.cuda.enable_flash_sdp(True)

IMAGE_FOLDER = '/home/yd344/dvornek_10t/Datasets/SA-Med2D/raw/MeCoVQA/SAMed2Dv1/'

def parse_json_to_conversations(json_file):
    data_list = json.load(open(json_file, 'r'))
    conversations = []
    GT_outputs = []
    message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
            ]
        },
    ]
    for entry in tqdm(data_list):
        text = entry["conversations"][0]['value']
        text = text.replace('<image>', '').replace('\n', '')
        gt_output = entry["conversations"][-1]['value']
        image_path = os.path.join(IMAGE_FOLDER, entry['image'])
        message = deepcopy(message_template)
        message[1]['content'][0]['text'] = text
        message[1]['content'][1]['image'] = image_path
        conversations.append(message)
        GT_outputs.append(gt_output)
    return conversations, GT_outputs


def save_outputs_to_json(outputs, filename, output_dir="./runs/output", model_info=None):
    """
    Save model outputs to a JSON file.
    
    Args:
        outputs (list): List of model outputs
        filename (str): Name of the output JSON file
        output_dir (str): Directory to save the outputs
        model_info (dict, optional): Additional model information to include
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Full path to the output file
    output_path = os.path.join(output_dir, filename)
    
    # Create a results dictionary with metadata
    results = {
        "outputs": outputs,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": len(outputs)
    }
    
    # Add model info if provided
    if model_info:
        results.update({"model_info": model_info})
    
    # Save the outputs
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Outputs saved to {output_path}")
    return output_path


def eval_medgemma(conversations, batch_size=2):
    model_id = "google/medgemma-4b-it"
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_id)
    outputs = []

    # Process in batches
    for i in tqdm(range(0, len(conversations), batch_size), desc="Processing MedGemma batches"):
        batch = conversations[i:i+batch_size]
        batch_inputs = []
        input_lens = []
        
        # Prepare batch inputs
        for messages in batch:
            image_path = messages[1]['content'][1]['image']
            image = Image.open(image_path)
            messages[1]['content'][1]['image'] = image
            
            input_data = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)
            
            input_lens.append(input_data["input_ids"].shape[-1])
            batch_inputs.append(input_data)
        
        # Batch processing
        with torch.inference_mode():
            for j, inputs in enumerate(batch_inputs):
                generation = model.generate(**inputs, max_new_tokens=500, do_sample=False)
                generation = generation[0][input_lens[j]:]
                
                decoded = processor.decode(generation, skip_special_tokens=True)
                outputs.append(decoded)
    
    return outputs

def eval_qwen_vl(conversations, batch_size=2):
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_id)
    outputs = []

    # Process in batches
    for i in tqdm(range(0, len(conversations), batch_size), desc="Processing Qwen-VL batches"):
        batch = conversations[i:i+batch_size]
        batch_inputs = []
        input_lens = []
        
        # Prepare batch inputs
        for messages in batch:
            image_path = messages[1]['content'][1]['image']
            image = Image.open(image_path)
            messages[1]['content'][1]['image'] = image
            
            input_data = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)
            
            input_lens.append(input_data["input_ids"].shape[-1])
            batch_inputs.append(input_data)
        
        # Batch processing
        with torch.inference_mode():
            for j, inputs in enumerate(batch_inputs):
                generation = model.generate(**inputs, max_new_tokens=500, do_sample=False)
                generation = generation[0][input_lens[j]:]
                
                decoded = processor.decode(generation, skip_special_tokens=True)
                outputs.append(decoded)
    
    return outputs

if __name__ == "__main__":
    # Create a timestamped directory for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./runs/output", f"eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    json_to_eval = 'data/MeCoVQA/test/MeCoVQA_Complex_VQA_test.json'
    conversations, gts = parse_json_to_conversations(json_to_eval)
    
    # Number of samples to evaluate
    num_samples = 5
    
    # Save evaluation configuration
    config = {
        "timestamp": timestamp,
        "dataset": json_to_eval,
        "num_samples": num_samples,
        "models": ["google/medgemma-4b-it", "Qwen/Qwen2.5-VL-7B-Instruct"]
    }
    config_path = os.path.join(output_dir, "eval_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Evaluate MedGemma
    print(f"Evaluating MedGemma on {num_samples} samples...")
    medgemma_outputs = eval_medgemma(conversations[:num_samples], batch_size=2)
    medgemma_model_info = {
        "model_name": "google/medgemma-4b-it",
        "model_type": "Image-Text-to-Text",
        "batch_size": 2
    }
    medgemma_output_path = save_outputs_to_json(
        medgemma_outputs, 
        "medgemma_outputs.json", 
        output_dir,
        medgemma_model_info
    )
    
    # Evaluate Qwen-VL
    print(f"Evaluating Qwen-VL on {num_samples} samples...")
    qwen_outputs = eval_qwen_vl(conversations[:num_samples], batch_size=2)
    qwen_model_info = {
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_type": "Image-Text-to-Text",
        "batch_size": 2
    }
    qwen_output_path = save_outputs_to_json(
        qwen_outputs, 
        "qwen_outputs.json", 
        output_dir,
        qwen_model_info
    )
    
    # Save ground truth outputs for reference
    gt_info = {
        "source": json_to_eval,
        "dataset": "MeCoVQA"
    }
    gt_output_path = save_outputs_to_json(
        gts[:num_samples], 
        "ground_truth.json", 
        output_dir,
        gt_info
    )
    
    # Print examples
    print("\n### MedGemma Output Example:")
    print(medgemma_outputs[0])
    
    print("\n### Qwen-VL Output Example:")
    print(qwen_outputs[0])
    
    print("\n### Ground Truth Example:")
    print(gts[0])
    
    print(f"\nAll outputs saved to: {output_dir}")


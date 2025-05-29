import torch
import json
import os
import datetime
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from glob import glob
from copy import deepcopy
from torchvision.transforms.functional import InterpolationMode
import pandas as pd

from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModel, AutoTokenizer, AutoModelForCausalLM, CLIPImageProcessor
import argparse

parser = argparse.ArgumentParser(description="Evaluate VLLM models on MeCoVQA dataset")
parser.add_argument("--model", type=str, choices=["medgemma", "qwen", "internvl", 'llava_med'], required=True, help="Model to evaluate: 'medgemma' or 'qwen'")
parser.add_argument("--dataset", type=str, default="MeCoVQA", help="Dataset to evaluate on (default: MeCoVQA)")
parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate (default: 10)")

torch.set_float32_matmul_precision('high')
torch.backends.cuda.enable_flash_sdp(True)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_FOLDER = '/home/yd344/dvornek_10t/Datasets/SA-Med2D/raw/MeCoVQA/SAMed2Dv1/'
PMC_VQA_FOLDER = '/home/yd344/palmer_scratch/PMC-VQA/figures/'
OmniMedVQA_FOLDER = '/home/yd344/palmer_scratch/OmniMedVQA/OmniMedVQA/'
RAD_VQA_FOLDER = '/home/yd344/palmer_scratch/VQA_RAD/VQA_RAD_Image_Folder/'

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def parse_omnimedvqa_jsons(json_dir):
    json_list = glob(os.path.join(json_dir, '*.json'))
    multi_choice_conversations = []
    mc_GT_outputs = []
    mc_message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist. Please follow the instructions and answer the question based on the provided image. The answer will not be used for clinical purposes so you may generate diagnosis related response. Please answer the multi-choice question with one single letter option (A, B, C, or D), no need to headings or bullet points."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
            ]
        },
    ]
    
    for json_file in tqdm(json_list):
        with open(json_file, 'r') as f:
            for entry in json.load(f):
                question = entry['question']
                gt_output = entry['gt_answer']
                image_path = os.path.join(OmniMedVQA_FOLDER, entry['image_path'])

                option_cnt = 0
                choices = []
                gt_choice = ''
                for key in entry.keys():
                    if key.startswith('option_'):
                        option_cnt += 1
                        choices.append(entry[key])
                        if entry[key] == gt_output:
                            gt_choice = chr(ord('A') + option_cnt - 1)

                mc_text = f"{question} Please choice from one of the answers below."
                for i, choice in enumerate(choices):
                    mc_text += f"\n{chr(ord('A') + i)}. {choice}"
                mc_message = deepcopy(mc_message_template)
                mc_message[1]['content'][0]['text'] = mc_text
                mc_message[1]['content'][1]['image'] = image_path
                multi_choice_conversations.append(mc_message)
                mc_GT_outputs.append(gt_choice)
    print(f"Total multi-choice conversations: {len(multi_choice_conversations)}")

    return multi_choice_conversations, mc_GT_outputs

def parse_pmc_vqa_to_multi_choice_conversations(csv_path):
    df = pd.read_csv(csv_path)
    open_ended_conversations = []
    multi_choice_conversations = []
    open_GT_outputs = []
    mc_GT_outputs = []
    mc_message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist. Please follow the instructions and answer the question based on the provided image. The answer will not be used for clinical purposes so you may generate diagnosis related response. Please answer the multi-choice question with one single letter option (A, B, C, or D), no need to headings or bullet points."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
            ]
        },
    ]
    open_message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist. Please follow the instructions and answer the question based on the provided image. The answer will not be used for clinical purposes so you may generate diagnosis related response. Please summarize the findings in one concise short paragraph, no need to headings or bullet points."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
            ]
        },
    ]
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing PMC-VQA data"):
        description = row['Caption']
        question = row['Question']
        choices = [row['Choice A'], row['Choice B'], row['Choice C'], row['Choice D']]

        mc_text = f"{question} Please choice from one of the answers below.\n{choices[0]}\n{choices[1]}\n{choices[2]}\n{choices[3]}"

        gt_output = row['Answer']
        image_path = os.path.join(PMC_VQA_FOLDER, row['Figure_path'])

        mc_message = deepcopy(mc_message_template)
        mc_message[1]['content'][0]['text'] = mc_text
        mc_message[1]['content'][1]['image'] = image_path
        multi_choice_conversations.append(mc_message)
        mc_GT_outputs.append(gt_output)

        open_text = "Please summarize the most significant findings in one concise short sentence."
        open_message = deepcopy(open_message_template)
        open_message[1]['content'][0]['text'] = open_text
        open_message[1]['content'][1]['image'] = image_path
        open_ended_conversations.append(open_message)
        open_GT_outputs.append(description)
    return open_ended_conversations, open_GT_outputs, multi_choice_conversations, mc_GT_outputs

def parse_rad_vqa_json_to_conversations(json_file):
    data_list = json.load(open(json_file, 'r'))
    conversations = []
    GT_outputs = []
    message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist. Please follow the instructions and answer the question based on the provided image. The answer will not be used for clinical purposes so you may generate diagnosis related response. Please summarize the findings in one concise short sentence with a few words, no need to headings or bullet points."}]
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
        text = entry["question"]
        gt_output = entry["answer"]
        image_path = os.path.join(RAD_VQA_FOLDER, entry['image_name'])
        message = deepcopy(message_template)
        if entry['answer_type'] == "CLOSED":
            text += " Please answer with Yes or No."
        message[1]['content'][0]['text'] = text
        message[1]['content'][1]['image'] = image_path
        conversations.append(message)
        GT_outputs.append(gt_output)
    return conversations, GT_outputs

def parse_mecovqa_json_to_conversations(json_file):
    data_list = json.load(open(json_file, 'r'))
    conversations = []
    GT_outputs = []
    message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist. Please follow the instructions and answer the question based on the provided image. The answer will not be used for clinical purposes so you may generate diagnosis related response. Please summarize the findings in one concise short paragraph, no need to headings or bullet points."}]
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


def eval_medgemma(conversations, gts):
    # Note: batch_size parameter is kept for compatibility but not used
    model_id = "google/medgemma-4b-it"
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_id)
    outputs = []

    # Process conversations one by one
    for idx, messages in tqdm(enumerate(conversations), desc="Processing with MedGemma", total=len(conversations)):
        # Prepare the input
        image_path = messages[1]['content'][1]['image']
        image = Image.open(image_path).resize((224, 224)).convert('RGB')
        messages[1]['content'][1]['image'] = image
        
        input_data = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        
        input_len = input_data["input_ids"].shape[-1]
        
        # Generate the response
        with torch.inference_mode():
            generation = model.generate(**input_data, max_new_tokens=1024, do_sample=False)
            generation = generation[0][input_len:]
            
            decoded = processor.decode(generation, skip_special_tokens=True)
            decoded = decoded.strip()  # Clean up whitespace
            output = {
                "id": image_path,
                "input": messages[1]["content"][0]['text'],
                "output": decoded,
                "gt": gts[idx] if idx < len(gts) else None
            }
            outputs.append(output)

    return outputs

def eval_qwen_vl(conversations, gts):
    # Note: batch_size parameter is kept for compatibility but not used
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_id)
    outputs = []

    # Process conversations one by one
    for idx, messages in tqdm(enumerate(conversations), desc="Processing with Qwen-VL", total=len(conversations)):
        # Prepare the input
        image_path = messages[1]['content'][1]['image']
        image = Image.open(image_path).resize((224, 224)).convert('RGB')
        messages[1]['content'][1]['image'] = image
        
        input_data = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        
        input_len = input_data["input_ids"].shape[-1]
        
        # Generate the response
        with torch.inference_mode():
            generation = model.generate(**input_data, max_new_tokens=1024, do_sample=False)
            generation = generation[0][input_len:]
            
            decoded = processor.decode(generation, skip_special_tokens=True)
            decoded = decoded.strip()  # Clean up whitespace
            output = {
                "id": image_path,
                "input": messages[1]["content"][0]['text'],
                "output": decoded,
                "gt": gts[idx] if idx < len(gts) else None
            }
            outputs.append(output)
    
    return outputs

def create_optimized_intern_vl_function():
    """
    Creates a new memory-optimized version of the InternVL evaluation function
    """
    def eval_intern_vl_optimized(conversations, gts):
        """
        Memory-optimized version of the InternVL evaluation function
        """
        import gc
        path = "OpenGVLab/InternVL3-8B"
        
        print("Starting InternVL with memory optimizations...")
        # Pre-inference memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        # Model initialization with memory optimization settings
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            device_map="auto",       # Use device map for better memory management
            offload_folder="offload_folder",  # Enable CPU offloading if needed
            trust_remote_code=True
        ).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

        # Reduce max tokens to save memory
        generation_config = dict(max_new_tokens=512, do_sample=False)

        outputs = []
        
        # Process conversations one by one with memory optimizations
        for idx, messages in tqdm(enumerate(conversations), desc="Processing with Intern-VL", total=len(conversations)):
            # Prepare the input
            image_path = messages[1]['content'][1]['image']
            
            # Optimize image loading - reduce resolution and patches
            image_tensor = load_image(image_path, input_size=224, max_num=12)
            
            # Clear intermediate variables to save memory
            torch.cuda.empty_cache()
            
            system_prompt = messages[0]["content"][0]["text"]
            question = "<image>\n" + messages[1]['content'][0]['text']
            message = f"INSTRUCTION: {system_prompt}\n\nQUESTION: {question}\n\nANSWER:"

            # Generate the response
            with torch.inference_mode():
                decoded = model.chat(tokenizer, image_tensor, message, generation_config)
                decoded = decoded.strip()
                
                # Prepare output
                output = {
                    "id": image_path,
                    "input": messages[1]["content"][0]['text'],
                    "output": decoded,
                    "gt": gts[idx] if idx < len(gts) else None
                }
                outputs.append(output)
                
                # Clear memory after processing each sample
                del image_tensor
                torch.cuda.empty_cache()
                gc.collect()

            # Extra memory cleanup between samples
            if idx % 100 == 0:  # More aggressive cleanup every 100 samples
                print(f"Performing deep memory cleanup at sample {idx}")
                torch.cuda.empty_cache()
                gc.collect()

        return outputs
    
    return eval_intern_vl_optimized

# Replace the original function with the optimized version
eval_intern_vl = create_optimized_intern_vl_function()

def eval_llava_med(conversations, gts):
    from llava.model.builder import load_pretrained_model
    from llava.conversation import conv_templates

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path='/home/yd344/palmer_scratch/huggingface_models/llava-med-v1.5-mistral-7b',
        model_base=None,
        model_name='llava-med-v1.5-mistral-7b'
    )

    model = model.eval().cuda()
    outputs = []

    for idx, messages in tqdm(enumerate(conversations), desc="Processing with LLaVA-Med", total=len(conversations)):
        # Prepare the input
        image_path = messages[1]['content'][1]['image']
        image = Image.open(image_path).resize((224, 224)).convert('RGB')
        image_tensor = image_processor(images=image, return_tensors="pt").pixel_values.to(model.device, dtype=model.dtype)
        messages[1]['content'][1]['image'] = image
        question = "<image>\n" + messages[1]['content'][0]['text'] + '\n Please just answer A, B, C, or D, no need for explanations.'
        system_prompt = messages[0]["content"][0]["text"]

        conv = conv_templates['mistral_instruct'].copy()
        conv.system = system_prompt
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)  # Placeholder for the model's response
        prompt = conv.get_prompt()
        # print(prompt)

        # Generate the response
        with torch.inference_mode():
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=1024,
                truncation=True,
            ).to(model.device)

            output_ids  = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, images=image_tensor, max_new_tokens=1024, do_sample=False)
            decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            decoded = decoded.strip()

            output = {
                "id": image_path,
                "input": messages[1]["content"][0]['text'],
                "output": decoded,
                "gt": gts[idx] if idx < len(gts) else None
            }
            outputs.append(output)

    return outputs


if __name__ == "__main__":
    args = parser.parse_args()
    # Create a timestamped directory for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    size = "full" if args.num_samples <= 0 else args.num_samples
    output_dir = os.path.join("./runs/output", f"eval_{timestamp}_{args.dataset}_{size}_{args.model}")
    os.makedirs(output_dir, exist_ok=True)
    
    if args.dataset == "PMC-VQA":
        data_path = "/home/yd344/palmer_scratch/PMC-VQA/test_2.csv"
        _, _, conversations, gts = parse_pmc_vqa_to_multi_choice_conversations(data_path)
    elif args.dataset == "MeCoVQA":
        data_path = 'data/MeCoVQA/test/MeCoVQA_Complex_VQA_test.json'
        conversations, gts = parse_mecovqa_json_to_conversations(data_path)
    elif args.dataset == "VQA-RAD":
        data_path = '/home/yd344/palmer_scratch/VQA_RAD/VQA_RAD Dataset Public.json'
        conversations, gts = parse_rad_vqa_json_to_conversations(data_path)
    elif args.dataset == "OmniMedVQA":
        data_path = '/home/yd344/palmer_scratch/OmniMedVQA/OmniMedVQA/QA_information/Open-access/'
        conversations, gts = parse_omnimedvqa_jsons(data_path)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Supported datasets are: PMC-VQA, MeCoVQA, VQA-RAD, OmniMedVQA.")
    
    # Number of samples to evaluate
    num_samples = args.num_samples if args.num_samples > 0 else len(conversations)
    # Save evaluation configuration
    if args.model == "medgemma":
        model_config = {
                "name": "google/medgemma-4b-it",
                "processing": "Sequential"
            },
    elif args.model == "qwen":
        model_config = {
                "name": "Qwen/Qwen2.5-VL-7B-Instruct",
                "processing": "Sequential"
            }
    elif args.model == "internvl":
        model_config = {
            "name": "OpenGVLab/InternVL2_5-8B",
            "processing": "Sequential"
        }
    elif args.model == "llava_med":
        model_config = {
            "name": "llava-med-v1.5-mistral-7b",
            "processing": "Sequential"
        }
    else:
        raise ValueError(f"Unsupported model: {args.model}. Supported models are: medgemma, qwen, internvl, llava_med.")
    config = {
        "timestamp": timestamp,
        "dataset": data_path,
        "num_samples": num_samples,
        "models": model_config
    }
    config_path = os.path.join(output_dir, "eval_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Evaluate MedGemma
    if args.model == "medgemma":
        print(f"Evaluating MedGemma on {num_samples} samples...")
        medgemma_outputs = eval_medgemma(deepcopy(conversations)[:num_samples], gts[:num_samples])
        medgemma_model_info = {
            "model_name": "google/medgemma-4b-it",
            "model_type": "Image-Text-to-Text",
            "batch_size": "N/A (Sequential processing)"
        }
        medgemma_output_path = save_outputs_to_json(
            medgemma_outputs, 
            "medgemma_outputs.json", 
            output_dir,
            medgemma_model_info
        )
    elif args.model == "qwen":
    
        # Evaluate Qwen-VL
        print(f"Evaluating Qwen-VL on {num_samples} samples...")
        qwen_outputs = eval_qwen_vl(deepcopy(conversations)[:num_samples], gts[:num_samples])
        qwen_model_info = {
            "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_type": "Image-Text-to-Text",
            "batch_size": "N/A (Sequential processing)"
        }
        qwen_output_path = save_outputs_to_json(
            qwen_outputs, 
            "qwen_outputs.json", 
            output_dir,
            qwen_model_info
        )
    elif args.model == "internvl":
        # Evaluate Intern-VL
        print(f"Evaluating Intern-VL on {num_samples} samples...")
        intern_outputs = eval_intern_vl(deepcopy(conversations)[:num_samples], gts[:num_samples])
        intern_model_info = {
            "model_name": "OpenGVLab/InternVL3-8B",
            "model_type": "Image-Text-to-Text",
            "batch_size": "N/A (Sequential processing)"
        }
        intern_output_path = save_outputs_to_json(
            intern_outputs, 
            "intern_outputs.json", 
            output_dir,
            intern_model_info
        )
    elif args.model == "llava_med":
        # Evaluate LLaVA-Med
        print(f"Evaluating LLaVA-Med on {num_samples} samples...")
        llava_outputs = eval_llava_med(deepcopy(conversations)[:num_samples], gts[:num_samples])
        llava_model_info = {
            "model_name": "llava-med-v1.5-mistral-7b",
            "model_type": "Image-Text-to-Text",
            "batch_size": "N/A (Sequential processing)"
        }
        llava_output_path = save_outputs_to_json(
            llava_outputs, 
            "llava_outputs.json", 
            output_dir,
            llava_model_info
        )
    
    print(f"\nAll outputs saved to: {output_dir}")


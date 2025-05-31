#!/bin/bash
# Merge LoRA weights and save the Hugging Face model

module load NCCL/2.16.2-GCCcore-12.2.0-CUDA-11.8.0
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX

python merge_lora_weights_and_save_hf_model.py --version "microsoft/llava-med-v1.5-mistral-7b"  --vision_pretrained "/home/yd344/palmer_scratch/huggingface_models/sam-med2d_b.pth"  --weight runs/medplib-7b-stage2_all/last_ckpt_model/global_step9261/mp_rank_00_model_states.pt --save_path runs/medplib-7b-stage2_all/hf
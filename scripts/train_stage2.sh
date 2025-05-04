
module load NCCL/2.16.2-GCCcore-12.2.0-CUDA-11.8.0
NCCL_DEBUG=WARN
time=$(date +%Y-%m-%d-%H-%M-%S)
exp_name="medplib-7b-stage2"
exp_dir="runs/$exp_name"
mkdir -p "$exp_dir"
# deepspeed --include=localhost:0, --master_port=65001 
python train_ds_medplib.py \
  --version="microsoft/llava-med-v1.5-mistral-7b" \
  --vision_tower='openai/clip-vit-large-patch14-336' \
  --data_path='/home/yd344/dvornek_10t/Datasets/MeCoVQA/train/MeCoVQA-Complex.json' \
  --val_data_path='/home/yd344/dvornek_10t/Datasets/MeCoVQA/test/MeCoVQA_Complex_VQA_test.json' \
  --image_folder='/home/yd344/dvornek_10t/Datasets/SA-Med2D/raw/MeCoVQA/SAMed2Dv1' \
  --vision_pretrained="/home/yd344/palmer_scratch/huggingface_models/sam-med2d_b.pth" \
  --exp_name=$exp_name \
  --epochs=3 \
  --batch_size=16 \
  --workers=8 \
  --image_aspect_ratio='pad' \
  --is_multimodal=True \
  --model_max_length 2048 \
  --grad_accumulation_steps 2 \
  --out_dim 256 \
  --ce_loss_weight 1.0 \
  --dice_loss_weight 5.0 \
  --bce_loss_weight 1.0 \
  --lora_r 16 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --sft_modules "lm_head,embed_tokens,input_layernorm,post_attention_layernorm,mm_projector" \
  --lr 0.0001 \
  --no_eval \
  --save_steps 400 \
  2>&1|tee -a runs/$exp_name/$time.log

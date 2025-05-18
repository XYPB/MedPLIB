
time=$(date +%Y-%m-%d-%H-%M-%S)
NCCL_DEBUG=WARN
exp_name="medplib-7b-stage3"
exp_dir="runs/$exp_name"
mkdir -p "$exp_dir"

deepspeed --include=localhost:0,1 --master_port=65000 train_ds_medplib.py \
  --version="/home/yd344/project/MedPLIB/runs/medplib-7b-stage2/hf" \
  --vision_tower='openai/clip-vit-large-patch14-336' \
  --data_path='/home/yd344/dvornek_10t/Datasets/MeCoVQA/train/MeCoVQA-Grounding.json' \
  --val_data_path='/home/yd344/dvornek_10t/Datasets/MeCoVQA/test/MeCoVQA_Grounding_test.json' \
  --image_folder='/home/yd344/dvornek_10t/Datasets/SA-Med2D/raw/MeCoVQA/SAMed2Dv1' \
  --vision_pretrained="/home/yd344/palmer_scratch/huggingface_models/sam-med2d_b.pth" \
  --exp_name=$exp_name \
  --epochs=10 \
  --batch_size=32 \
  --workers=16 \
  --image_aspect_ratio='pad' \
  --is_multimodal=True \
  --model_max_length 2048 \
  --grad_accumulation_steps 1 \
  --out_dim 256 \
  --ce_loss_weight 1.0 \
  --dice_loss_weight 5.0 \
  --bce_loss_weight 1.0 \
  --iou_loss_weight 0 \
  --focal_loss_weight 1.0 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_target_modules "gate_proj,up_proj,down_proj" \
  --sft_modules "mask_decoder,text_hidden_fcs" \
  --lr 0.0003 \
  --save_steps 300 \
  --sam_img_size 256 \
  --train_mask_decoder \
  --eval_only \
  2>&1|tee -a runs/$exp_name/$time.log

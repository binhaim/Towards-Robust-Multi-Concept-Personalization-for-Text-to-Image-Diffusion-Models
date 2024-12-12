# Kyunghee University Bachelor's Thesis
2021103746 Habin LIM

## Towards Robust Multi-Concept Personalization for Text-to-Image Diffusion Models

![Main Figure 2](https://github.com/user-attachments/assets/256afb10-391b-4296-b980-43703205f2b6)

---

## Dataset

### CustomDiffusion Dataset  
[Custom Diffusion GitHub Repository](https://github.com/adobe-research/custom-diffusion?tab=readme-ov-file)

### DreamBooth Dataset  
[DreamBooth GitHub Repository](https://github.com/google/dreambooth)

---

## Environment Setup

```bash
conda env create -f environment.yml
```

## Training
```
!accelerate launch --num_processes=1 ./RMP_Training.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --target_prompt=$TARGET_PROMPT \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --target_modules attn2.to_v \
  --weight_name="attn2_to_v.safetensors" \
  --checkpointing_steps 10
```
### Parameter Descriptions

- **`MODEL_NAME`**  
  The name of the base model you wish to use.  

- **`INSTANCE_DIR`**  
  The directory containing your training dataset.  

- **`OUTPUT_DIR`**  
  The directory where your trained LoRA and checkpoint files will be saved.  

- **`TARGET_PROMPT`**  
  The prompt you want to personalize.

## Inference
See the RMP_inference.ipynb

# Thanks To

This Project was inspired by and built upon the following resources:
- https://github.com/yuval-alaluf/Attend-and-Excite 
- https://github.com/huggingface/diffusers

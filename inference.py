from typing import List, Dict
import torch
import torch.nn as nn

import sys 
from pipeline import RMPPipeline
from config import RunConfig
from run import run_on_prompt, get_indices_to_alter
from utils.ptp_utils2 import AttentionStore
from utils import vis_utils
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from utils.ptp_utils import aggregate_attention

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from diffusers import StableDiffusionPipeline
from util import *

from safetensors.torch import load_file

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Custom LoRA 레이어 정의
class CustomLoRALayer(nn.Module):
    def __init__(self, base_layer, lora_As, lora_Bs, *args, **kwargs):
        super(CustomLoRALayer, self).__init__()
        self.concept_num = len(lora_As)
        self.base_layer = base_layer
        
        # lora_As와 lora_Bs를 ModuleDict에 nn.Linear로 저장
        self.lora_As = nn.ModuleDict(lora_As)
        self.lora_Bs = nn.ModuleDict(lora_Bs)

        self.idx = []

    def forward(self, x):
        # 원래 base_layer를 통한 계산
        base_output = self.base_layer(x)
        lora_outputs = []
        # LoRA 계산
        for idx, key in zip(self.idx, self.lora_As):
            lora_A_output = self.lora_As[key](x[:,idx,:])  # nn.Linear로 forward 적용
            lora_B_output = self.lora_Bs[key](lora_A_output)  # nn.Linear로 forward 적용
            lora_outputs.append(lora_B_output)
        for idx,lora_output in zip(self.idx,lora_outputs):
            base_output[:,idx,:] += lora_output
        
        return base_output

def replace_lora_layers(stable, LoRAs):
    # attn2.to_v의 LoRA 레이어를 CustomLoRALayer로 교체
    for name, module in stable.unet.named_modules():
        if name.endswith("attn2.to_v"):
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = dict(stable.unet.named_modules())[parent_name]

            # 새로운 CustomLoRALayer 생성
            lora_As = {}
            lora_Bs = {}

            for LoRA in LoRAs:
                # 여기서 LoRA는 하나의 concept에 대한 LoRA
                for key in LoRA.keys():
                    if name in key:
                        # down weight는 lora_As에, up weight는 lora_Bs에 저장
                        if "lora_A" in key:
                            down_weight = LoRA[key]
                            # nn.Linear로 변환 (입력 차원과 출력 차원을 자동 추론)
                            in_features = down_weight.shape[1]
                            out_features = down_weight.shape[0]
                            lora_As[f'{len(lora_As)}'] = nn.Linear(in_features, out_features, bias=False)
                            lora_As[f'{len(lora_As)-1}'].weight = nn.Parameter(down_weight)
                        elif "lora_B" in key:
                            up_weight = LoRA[key]
                            in_features = up_weight.shape[1]
                            out_features = up_weight.shape[0]
                            lora_Bs[f'{len(lora_Bs)}'] = nn.Linear(in_features, out_features, bias=False)
                            lora_Bs[f'{len(lora_Bs)-1}'].weight = nn.Parameter(up_weight)

            print(lora_As)
            print(lora_Bs)

            new_module = CustomLoRALayer(
                base_layer=module,
                lora_As=lora_As,
                lora_Bs=lora_Bs,
            )

            # setattr을 사용하여 실제 모듈 교체
            setattr(parent_module, child_name, new_module)
            print(f"Changed {name} from {module.__class__.__name__} to {new_module.__class__.__name__}")
    
    return stable

def update_custom_layer_indices(stable, new_indices):
    for name, module in stable.unet.named_modules():
        # Check if the module is an instance of CustomLoRALayer
        if isinstance(module, CustomLoRALayer):
            module.idx = new_indices  # Update the idx attribute
            print(f"Updated idx for {name} to {new_indices}")
            
def run_and_display(prompts: List[str],
                    controller: AttentionStore,
                    indices_to_alter: List[int],
                    generator: torch.Generator,
                    target_controller: AttentionStore = None,
                    run_standard_sd: bool = False,
                    scale_factor: int = 20,
                    thresholds: Dict[int, float] = {0: 0.05, 10: 0.5, 20: 0.8},
                    max_iter_to_alter: int = 25,
                    n_inference_steps = 50,
                    display_output: bool = False):
    config = RunConfig(prompt=prompts[0],
                       run_standard_sd=run_standard_sd,
                       scale_factor=scale_factor,
                       thresholds=thresholds,
                       max_iter_to_alter=max_iter_to_alter,
                       n_inference_steps= n_inference_steps)
    image = run_on_prompt(model=stable,
                          prompt=prompts,
                          controller=controller,
                          target_controller=target_controller,
                          token_indices=indices_to_alter,
                          seed=generator,
                          config=config)
    if display_output:
        display(image)
    return image


def main():
    parser = argparse.ArgumentParser(description="Run Stable Diffusion with custom LoRA layers.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for generation")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--output_dir", type=str, default="output_images", help="Directory to save generated images")
    parser.add_argument("--model_paths", nargs="+", required=True, help="Paths to LoRA model checkpoints")

    args = parser.parse_args()

    # Seed 설정
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 모델 및 tokenizer 설정
    MODEL_NAME = "runwayml/stable-diffusion-v1-5"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    stable = AttendAndExcitePipeline.from_pretrained(MODEL_NAME).to(device)
    tokenizer = stable.tokenizer

    # LoRA 모델 로드 및 적용
    loras = [load_file(path) for path in args.model_paths]
    stable = replace_lora_layers(stable, loras)

    # 이미지 저장 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # Attention Store 및 레이어 인덱스 설정
    controller = AttentionStore()
    layer_indices = [5, 10]
    update_custom_layer_indices(stable, layer_indices)

    # 이미지 생성
    generator = torch.Generator('cuda').manual_seed(args.seed)
    image = run_and_display(prompts=[args.prompt],
                            controller=controller,
                            indices_to_alter=layer_indices,
                            generator=generator,
                            display_output=True)

    # 이미지 저장
    image_path = os.path.join(args.output_dir, f"{args.seed}.png")
    image.save(image_path)
    print(f"Image saved to {image_path}")

if __name__ == "__main__":
    main()
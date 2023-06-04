import random

import einops
import numpy as np
import torch
import cv2
import os
from ldm_hacked import DDIMSampler
from ldm_hacked import create_model, load_state_dict, DDIMSampler
from pytorch_lightning import seed_everything

# ----------------------- #
#   使用的参数
# ----------------------- #
# config的地址
config_path = "model_data/sd_v15.yaml"
# 模型的地址
model_path  = "model_data/v1-5-pruned-emaonly.safetensors"

# 生成的图像大小为input_shape
input_shape = [512, 512]
# 一次生成几张图像
num_samples = 2
# 采样的步数
ddim_steps  = 20
# 采样的种子，为-1的话则随机。
seed        = 12345
# eta
eta         = 0

# 提示词
prompt      = "a cat"
# 正面提示词
a_prompt    = "best quality, extremely detailed"
# 负面提示词
n_prompt    = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
# 正负扩大倍数
scale       = 9

# save_path
save_path   = "imgs/outputs_imgs"

# ----------------------- #
#   创建模型
# ----------------------- #
model   = create_model(config_path).cpu()
model.load_state_dict(load_state_dict(model_path, location='cuda'), strict=False)
model   = model.cuda()
ddim_sampler = DDIMSampler(model)

with torch.no_grad():
    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    # ----------------------- #
    #   获得编码后的prompt
    # ----------------------- #
    cond    = {"c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
    un_cond = {"c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
    H, W    = input_shape
    shape   = (4, H // 8, W // 8)

    # ----------------------- #
    #   进行采样
    # ----------------------- #
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)

    # ----------------------- #
    #   进行解码
    # ----------------------- #
    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

# ----------------------- #
#   保存图片
# ----------------------- #
if not os.path.exists(save_path):
    os.makedirs(save_path)
for index, image in enumerate(x_samples):
    cv2.imwrite(os.path.join(save_path, str(index) + ".jpg"), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

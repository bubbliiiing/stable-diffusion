## Stable Diffusion模型在pytorch当中的实现
---

该库参考于https://github.com/CompVis/stable-diffusion  
删去了一些配置文件，与一些暂时无用的文件夹。并在必要的地方添加了一些中文注释，以便于理解。该库并未调整ldm文件夹的结构，以免产生与其它Stable Diffusion仓库的Gap。  
后续会陆续增加img2img、训练、controlnet等。  

## 目录
1. [仓库更新 Top News](#仓库更新)
2. [性能情况 Performance](#性能情况)
3. [所需环境 Environment](#所需环境)
4. [文件下载 Download](#文件下载)
5. [训练步骤 How2train](#训练步骤)
6. [预测步骤 How2predict](#预测步骤)
7. [参考资料 Reference](#Reference)

## Top News
**`2023-04`**:**仓库创建，支持Stable Diffusion的txt2img预测。**  

## 性能情况
参考stable-diffusion的论文哈。  
https://ommer-lab.com/research/latent-diffusion-models/

## 所需环境
torch==2.0.1   
推荐torch==2.0.1，大多stable diffusion**基于这个版本**，webui也是。
```
# 安装torch==2.0.1 
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# 安装其它requirement
pip install -r requirements.txt
# 为了加速可安装xformers。
pip install xformers==0.0.20
```

## 文件下载
训练所需的权值可在百度网盘中下载。  
链接: https://pan.baidu.com/s/1p4e1-jcJJt3lCFZeMpYbwA    
提取码: vry7     
  
Flickr8k数据集也可以在百度网盘中下载。  
链接：https://pan.baidu.com/s/1I2FfEOhcBOupUazJP18ADQ    
提取码：lx57   
训练需要较高的显存要求，需要20G左右。   

## 训练步骤
首先准备好训练数据集，数据集摆放格式为：
```
- datasets
  - train
    1.jpg
    2.jpg
    3.jpg
    4.jpg
    5.jpg
    .......
  - metadata.jsonl
```
metadata.jsonl中每一行代表一个样本，file_name代表样本的相对路径，text代表样本对应的文本。
```
{"file_name": "train/1000268201_693b08cb0e.jpg", "text": "A child in a pink dress is climbing up a set of stairs in an entry way ."}
```

可首先使用上述提供的Flickr8k数据集为例进行尝试。

## 预测步骤
### a、txt2img
1. 下载完库后解压，在百度网盘下载权值，放入model_data，运行predict.py.
2. 根据需求修改predict.py文件中的prompt以实现不同目标的生成。
```
# ----------------------- #
#   使用的参数
# ----------------------- #
# config的地址
config_path = "model_data/sd_v15.yaml"
# 模型的地址
model_path  = "model_data/v1-5-pruned-emaonly.safetensors"
# fp16，可以加速与节省显存
sd_fp16     = True
vae_fp16    = True

# ----------------------- #
#   生成图片的参数
# ----------------------- #
# 生成的图像大小为input_shape，对于img2img会进行Centter Crop
input_shape = [512, 512]
# 一次生成几张图像
num_samples = 1
# 采样的步数
ddim_steps  = 20
# 采样的种子，为-1的话则随机。
seed        = 12345
# eta
eta         = 0
# denoise强度，for img2img
denoise_strength = 1.00

# ----------------------- #
#   提示词相关参数
# ----------------------- #
# 提示词
prompt      = "a cute cat, with yellow leaf, trees"
# 正面提示词
a_prompt    = "best quality, extremely detailed"
# 负面提示词
n_prompt    = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
# 正负扩大倍数
scale       = 9
# img2img使用，如果不想img2img这设置为None。
image_path  = None
# inpaint使用，如果不想inpaint这设置为None；inpaint使用需要结合img2img。
# 注意mask图和原图需要一样大
mask_path   = None

# ----------------------- #
#   保存路径
# ----------------------- #
save_path   = "imgs/outputs_imgs"
```

### b、img2img 
1. 下载完库后解压，在百度网盘下载权值，放入model_data.
2. 修改其中的image_path与denoise_strength。
3. 根据需求修改predict.py文件中的prompt以实现不同目标的生成。
```
# ----------------------- #
#   使用的参数
# ----------------------- #
# config的地址
config_path = "model_data/sd_v15.yaml"
# 模型的地址
model_path  = "model_data/v1-5-pruned-emaonly.safetensors"
# fp16，可以加速与节省显存
sd_fp16     = True
vae_fp16    = True

# ----------------------- #
#   生成图片的参数
# ----------------------- #
# 生成的图像大小为input_shape，对于img2img会进行Centter Crop
input_shape = [512, 512]
# 一次生成几张图像
num_samples = 1
# 采样的步数
ddim_steps  = 20
# 采样的种子，为-1的话则随机。
seed        = 12345
# eta
eta         = 0
# denoise强度，for img2img
denoise_strength = 1.00

# ----------------------- #
#   提示词相关参数
# ----------------------- #
# 提示词
prompt      = "a cute cat, with yellow leaf, trees"
# 正面提示词
a_prompt    = "best quality, extremely detailed"
# 负面提示词
n_prompt    = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
# 正负扩大倍数
scale       = 9
# img2img使用，如果不想img2img这设置为None。
image_path  = "imgs/test_imgs/cat.jpg"
# inpaint使用，如果不想inpaint这设置为None；inpaint使用需要结合img2img。
# 注意mask图和原图需要一样大
mask_path   = None

# ----------------------- #
#   保存路径
# ----------------------- #
save_path   = "imgs/outputs_imgs"
```

### c、inpaint
1. 下载完库后解压，在百度网盘下载权值，放入model_data.
2. 修改其中的image_path，mask_path与denoise_strength。
3. 根据需求修改predict.py文件中的prompt以实现不同目标的生成。
```
# ----------------------- #
#   使用的参数
# ----------------------- #
# config的地址
config_path = "model_data/sd_v15.yaml"
# 模型的地址
model_path  = "model_data/v1-5-pruned-emaonly.safetensors"
# fp16，可以加速与节省显存
sd_fp16     = True
vae_fp16    = True

# ----------------------- #
#   生成图片的参数
# ----------------------- #
# 生成的图像大小为input_shape，对于img2img会进行Centter Crop
input_shape = [512, 512]
# 一次生成几张图像
num_samples = 1
# 采样的步数
ddim_steps  = 20
# 采样的种子，为-1的话则随机。
seed        = 12345
# eta
eta         = 0
# denoise强度，for img2img
denoise_strength = 1.00

# ----------------------- #
#   提示词相关参数
# ----------------------- #
# 提示词
prompt      = "a cute dog, with yellow leaf, trees"
# 正面提示词
a_prompt    = "best quality, extremely detailed"
# 负面提示词
n_prompt    = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
# 正负扩大倍数
scale       = 9
# img2img使用，如果不想img2img这设置为None。
image_path  = "imgs/test_imgs/cat.jpg"
# inpaint使用，如果不想inpaint这设置为None；inpaint使用需要结合img2img。
# 注意mask图和原图需要一样大
mask_path   = "imgs/test_imgs/cat_mask.jpg"

# ----------------------- #
#   保存路径
# ----------------------- #
save_path   = "imgs/outputs_imgs"
```

## Reference
https://github.com/lllyasviel/ControlNet   
https://github.com/CompVis/stable-diffusion  
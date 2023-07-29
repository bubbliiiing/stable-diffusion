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
torch==1.13.1     
推荐torch==1.13.1，毕竟其它stable diffusion大多**大于或者等于这个版本**。
```
pip install -r requirements.txt
# 为了加速可安装xformers，该版本与torch==1.13.1适配。
pip install xformers==0.0.16
```
当然也推荐使用torch>=2.0.0

## 文件下载
训练所需的权值可在百度网盘中下载。  
链接: 链接: https://pan.baidu.com/s/1p4e1-jcJJt3lCFZeMpYbwA    
提取码: vry7    

## 训练步骤
待办

## 预测步骤
### a、txt2img
1. 下载完库后解压，在百度网盘下载权值，放入model_data，运行predict.py.
2. 根据需求修改predict.py文件中的prompt以实现不同目标的生成。

### b、img2img 
1. 下载完库后解压，在百度网盘下载权值，放入model_data.
2. 修改其中的image_path与denoise_strength。
3. 根据需求修改predict.py文件中的prompt以实现不同目标的生成。

## Reference
https://github.com/lllyasviel/ControlNet   
https://github.com/CompVis/stable-diffusion  
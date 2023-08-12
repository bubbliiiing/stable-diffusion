import pytorch_lightning as pl
from share import *
from torch.utils.data import DataLoader

from ldm_hacked import *

if __name__ == '__main__':
    devices             = [0]
    # Config信息，这里是sd15
    config_path         = 'model_data/sd_v15.yaml'
    # 载入的权重路径
    model_path          = 'model_data/v1-5-pruned-emaonly.safetensors'
    # 训练目录
    train_data_dir      = 'datasets'
    # 输入的训练图片大小
    input_shape         = [512, 512]
    # 批次大小
    batch_size          = 2
    # 图片多久打印一次
    logger_freq         = 500
    # 学习率大小
    learning_rate       = 1e-5
    # 计算精度，16或者32
    precision           = 16
    # 数据加载时的num_workers数量
    num_workers         = 8

    # 首先加载模型
    model               = create_model(config_path).cpu()
    model.load_state_dict(load_state_dict(model_path, location='cpu'), strict=False)
    model.learning_rate = learning_rate

    # 然后加载数据集
    dataset     = MyDataset(train_data_dir, input_shape)
    dataloader  = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, pin_memory=True)

    # 加载trainer
    trainer     = pl.Trainer(devices=devices, precision=precision, callbacks=[ImageLogger(batch_frequency=logger_freq)])

    # 开始训练
    trainer.fit(model, dataloader)
import json
import cv2
import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

class CenterCrop(object):
    def __init__(self, input_shape = (512, 512)):
        self.height         = input_shape[0]
        self.width          = input_shape[1]
        self.origin_ratio   = float(self.height) / self.width

    def __call__(self, image):
        ratio       = self.width / self.height
        src_ratio   = image.width / image.height

        src_w = self.width if ratio > src_ratio else image.width * self.height // image.height
        src_h = self.height if ratio <= src_ratio else image.height * self.width // image.width

        resized = image.resize((src_w, src_h))
        res     = Image.new("RGB", (self.width, self.height))
        res.paste(resized, box = (self.width // 2 - src_w // 2, self.height // 2 - src_h // 2))

        return res
    
class MyDataset(Dataset):
    def __init__(self, datasets_path, input_shape):
        self.data           = []
        self.datasets_path  = datasets_path
        self.input_shape    = input_shape
        self.transforms     = CenterCrop()

        with open(os.path.join(datasets_path, 'metadata.jsonl'), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        target_filename = item['file_name']
        prompt          = item['text']

        target = Image.open(os.path.join(self.datasets_path, target_filename))
        target = self.transforms(target)

        # Normalize target images to [-1, 1].
        target = (np.array(target, np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt)
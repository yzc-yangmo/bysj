import os, json
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class FoodImageTransform:
    # 根据transform_code返回对应的transform
    def __init__(self, transform_code=0):
        self.transform_code = transform_code

    def __call__(self):
        
        # 方案0：统一缩放到512x512  
        if self.transform_code == 0:
            return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
            ])
        
        # 方案1：保持长宽比，短边填充
        elif self.transform_code == 1:
            return transforms.Compose([
            transforms.Resize(512),  # 短边缩放到512
            transforms.CenterCrop((512, 512)),
            transforms.ToTensor()
            ])
        
        else:
            raise ValueError('Invalid transform code')
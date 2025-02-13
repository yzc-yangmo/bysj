import os, json
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class FoodImageTransform:

    # 根据transform_code返回对应的transform
    def __init__(self, transform_type=0):
        
        if transform_type == 0:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])
        elif transform_type == 1:
            self.transform = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop((512, 512)),
                transforms.ToTensor()
            ])
        else:
            raise ValueError('Invalid transform code')
    
    def __call__(self, image):  # 接收图像参数
        return self.transform(image)  # 直接转换图像
        

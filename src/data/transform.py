import os, json
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class FoodImageTransform:

    # 根据transform_code返回对应的transform
    def __init__(self, transform_type = 0):
        
        if transform_type == 0:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        elif transform_type == 1:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop((256, 256)),
                transforms.ToTensor()
            ])
        else:
            raise ValueError('Invalid transform code')
    
    def __call__(self, image):  # 接收图像参数
        return self.transform(image)  # 直接转换图像
        

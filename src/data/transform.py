import os, json
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms import RandAugment

class FoodImageTransform:

    # 根据transform_code返回对应的transform
    def __init__(self, transform_type = 0):
        
        # 不进行数据增强，仅进行归一化
        if transform_type == 0:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        # 一般数据增强方案
        elif transform_type == 1:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        # 添加更强的数据增强方案
        elif transform_type == 2:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.4, scale=(0.1, 0.5))
            ])
            
        else:
            raise ValueError('Invalid transform code')
    
    def __call__(self, image):  # 接收图像参数
        return self.transform(image)  # 直接转换图像
        

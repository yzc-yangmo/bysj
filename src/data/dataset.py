import os, json
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoade

from transform import FoodImageTransform

# 定义食物数据集
class FoodImageDataset(Dataset):
    
    def __init__(self, dataset_path, transform=FoodImageTransform(1)): # 默认使用transform方案1
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    
    # Implement __len__
    def __len__(self):
        return len(self.image_files)
    
    # Implement __getitem__
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        # 获取图片的张量表示
        image = Image.open(image_path).convert('RGB')
        image_tensor  = self.transform(image) # 调用transform的__call__方法
        
        # 根据路径获取标签
        label = os.path.basename(image_path).split('_')[1]
        
            
        return image_tensor, label
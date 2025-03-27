import os, json
from PIL import Image
import torch
from torch.utils.data import Dataset
from .transform import FoodImageTransform



config = json.load(open('./config.json'))
mapping = json.load(open('./mapping.json'))

# 定义食物数据集
class FoodImageDataset(Dataset):
    
    def __init__(self, dataset_path, transform_type=1):
        self.dataset_path = dataset_path
        self.transform = FoodImageTransform(transform_type)
        self.image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]
        self.mapping = mapping if config["train"]["num_classes"] == 101 else dict(zip(set(i.split("-")[1] for i in os.listdir(config["train"]["dataset"]["train_path"])), range(config["train"]["num_classes"])))
    
    # Implement __len__
    def __len__(self):
        return len(self.image_files)
    
    # Implement __getitem__
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        # 获取图片的张量表示
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image) # 调用transform的__call__方法
        
        # 根据路径获取标签,数据集图片命名格式 img-label-id.jpg
        filename = os.path.basename(image_path)
        if len(filename.split("-")) != 3: 
            raise ValueError('Invalid filename format')
        
        label = filename.split('-')[1]
        
        # 获取张量表示的标签
        num_clsses = config["train"]["num_classes"]
        label_tensor = torch.zeros(num_clsses)
        label_tensor[self.mapping[label]] = 1
        
        return image_tensor, label_tensor
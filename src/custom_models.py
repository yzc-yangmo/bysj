import os, json
import torch
import torch.nn as nn
from torchvision import models




# FoodRecognitionModel: FRM
class FRM_20250213_1(nn.Module):
    
    # define the neural network structure
    def __init__(self):
        super().__init__()
        
        self.num_classes = json.load(open('../config/config.json'))["model"]["num_classes"] # 读取配置文件获取分类数
        self.resnet50 = models.resnet50() # 使用未预训练的resnet50模型
        
        num_features = self.resnet50.fc.in_features
        
        # 添加一个卷积层来降低特征图尺寸
        self.conv = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        
        
        self.resnet50.fc = nn.Sequential(  # 修改最后的全连接层以匹配类别数
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )
        
    # define the forward pass
    def forward(self, x):
        # 512x512x3 -> 256x256x3 -> 128x128x3  
        x = self.conv(x)
        x = self.resnet50(x)
        return x

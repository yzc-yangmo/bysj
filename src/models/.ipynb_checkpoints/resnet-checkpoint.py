import os, json
import torch
import torch.nn as nn
from torchvision import models


config = json.load(open('./config.json'))

num_classes = config["model"]["num_classes"]
drop_rate = json.load(open('./config.json'))["model"]["drop_rate"]

# FoodRecognitionModel: FRM
class resnet_v1(nn.Module):
    def __init__(self, num_classes=num_classes, pretrained=True, drop_rate=drop_rate):
        super().__init__()
        self.num_classes = num_classes
        
        # 自定义卷积层序列
        self.conv = nn.Sequential(
            # 输出：256x256x64
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            # 输出：128x128x64
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU()
        )
        
        # ResNet50主干网络
        self.resnet50 = models.resnet50(pretrained=pretrained)
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, self.num_classes)
        )
        
    # define the forward pass
    def forward(self, x):
        x = self.conv(x)  # 先通过自定义卷积层
        x = self.resnet50(x)  # 再通过ResNet50
        return x

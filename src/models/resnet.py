import os
import json
import torch
import torch.nn as nn
from torchvision import models

# 加载配置文件
config = json.load(open('./config.json'))

num_classes = config["model"]["num_classes"]
drop_rate = config["model"]["drop_rate"]

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

        # 自适应池化层，将特征图尺寸调整为 224x224
        self.pool = nn.AdaptiveAvgPool2d((224, 224))

        # ResNet50主干网络
        self.resnet50 = models.resnet50(pretrained=pretrained)
        # 重新定义第一层卷积，使其输入通道数为 64
        self.resnet50.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
        x = self.pool(x)  # 再通过自适应池化层
        x = self.resnet50(x)  # 最后通过ResNet50
        return x

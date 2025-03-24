import json
import torch.nn as nn
from torchvision import models

# 加载配置文件

class resnet_v1(nn.Module):
    def __init__(self, num_classes, drop_rate, pretrained=False):
        super(resnet_v1, self).__init__()
        
        # 加载预训练的ResNet-50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # 获取原始全连接层的输入特征维度
        num_ftrs = self.resnet.fc.in_features
        
        # 替换全连接层，添加Dropout
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)
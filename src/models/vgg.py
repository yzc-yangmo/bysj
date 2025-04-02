import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从配置文件读取参数
num_classes = json.load(open('./config.json'))["train"]["num_classes"]
drop_rate = json.load(open('./config.json'))["train"]["drop_rate"]

# VGG网络配置，数字表示输出通道，'M'表示最大池化层
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],  # VGG11
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],  # VGG13
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],  # VGG16
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],  # VGG19
}

class VGG(nn.Module):
    """
    VGG网络模型类，支持多种VGG变体
    
    Args:
        variant (str): VGG变体，可选 'vgg11', 'vgg13', 'vgg16', 'vgg19'
        batch_norm (bool): 是否使用批量归一化，默认为True
        num_classes (int): 分类类别数，默认从配置文件获取
        drop_rate (float): Dropout比率，默认从配置文件获取
        init_weights (bool): 是否初始化权重，默认为True
    """
    def __init__(self, variant='vgg16', batch_norm=True, num_classes=num_classes, 
                 drop_rate=drop_rate, init_weights=True):
        super(VGG, self).__init__()
        
        # 根据变体选择配置
        variant_map = {
            'vgg11': 'A',
            'vgg13': 'B',
            'vgg16': 'D',
            'vgg19': 'E'
        }
        
        if variant not in variant_map:
            raise ValueError(f"不支持的VGG变体: {variant}，支持的变体有: {list(variant_map.keys())}")
        
        cfg_key = variant_map[variant]
        
        # 构建特征提取层
        self.features = self._make_layers(cfgs[cfg_key], batch_norm)
        
        # 自适应平均池化确保输出大小为7x7
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(drop_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(drop_rate),
            nn.Linear(4096, num_classes),
        )
        
        if init_weights:
            self._initialize_weights()
    
    def _make_layers(self, cfg, batch_norm=True):
        """构建VGG的特征提取部分"""
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0) 
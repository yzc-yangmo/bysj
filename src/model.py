import os, json
import torch
from models import vit, resnet, vgg

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self.set_model()
    
    # 根据模型名称设置模型
    def set_model(self):
        # 包括：vit, resnet, vgg
        if self.model_name == "vit":
            return vit.VisionTransformer()
        elif self.model_name == "resnet":
            return resnet.ResNet()
        elif self.model_name == "vgg":
            return vgg.VGG()
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")
    
    # 获取模型
    def get_model(self):
        return self.model
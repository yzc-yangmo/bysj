import os, json
import torch
from models import vit, resnet

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self.set_model()
    
    # 根据模型名称设置模型
    def set_model(self):
        if self.model_name == "vit":
            return vit.VisionTransformer()
        elif self.model_name == "resnet":
            return resnet.ResNet()
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")
    
    # 获取模型
    def get_model(self):
        return self.model

    # 加载模型
    def load_model_state(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    
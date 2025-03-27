import torch
from PIL import Image
from models import vit, resnet
from data import transform

class Inference:
    def __init__(self, device):
        self.device = device

    # 加载模型
    def load_model(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()
        self.model.to(self.device)
    
    # 图像预处理
    def transform(self, image):
        transform = transform.FoodImageTransform(transform_type=0)
        image = transform(image)
        return image
    
    # 预测，返回mapping.json中的索引
    def predict(self, image_path):
        image = Image.open(image_path)
        image = self.transform(image)
        image = image.to(self.device)
        output = self.model(image)
        # softmax
        return output
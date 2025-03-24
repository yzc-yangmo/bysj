from models import vit, resnet

class Model():
    def __init__(self, model_name, num_classes, drop_rate):
        self.model_name = model_name
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        
        if model_name == "vit":
            self.model = vit.VisionTransformer(num_classes, drop_rate)
        elif model_name == "resnet":
            self.model = resnet.ResNet(num_classes, drop_rate)
        else:
            raise ValueError(f"不支持的模型: {model_name}")

    def __str__(self):
        return f"Model(model_name={self.model_name}, num_classes={self.num_classes}, drop_rate={self.drop_rate})"


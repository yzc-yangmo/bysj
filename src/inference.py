import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Inference:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def inference(self, image):
        self.model.eval()
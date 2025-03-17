import os, json, time
from datetime import datetime
from PIL import Image
import numpy as np
import pandas as pd
import wandb
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import models 
from vit_model import VisionTransformer

from data.dataset import FoodImageDataset
import custom_models

# 读取配置文件
config = json.load(open('./config.json'))
mapping = json.load(open('./mapping.json'))


# 配置wandb
wandb.init(project="food-image-classification（bysj）", 
           name=f"vit-demo-{time.strftime('%Y%m%d-%H%M%S')}",
           config=config)
wandb_log = {}

# 打印超参数
print(f"----------------config----------------")
for k, v in config.items():
    print(k)
    for kk, vv in v.items():
        print(f"--{kk}: {vv}")
print(f"--------------------------------")


# 读取数据集
train_foodimages = FoodImageDataset(config["dataset"]["train_path"])
val_foodimages = FoodImageDataset(config["dataset"]["val_path"])
train_loader = DataLoader(train_foodimages, batch_size=config["train"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_foodimages, batch_size=config["train"]["batch_size"], shuffle=True)

# 训练函数
def train_model(model, train_loader, val_loader):
    # 根据配置文件定义超参数
    lr = config["train"]["lr"]
    num_epochs = config["train"]["num_epochs"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device}!")
    
    # 移动模型到指定设备
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    
    print(f"trian start!")
    for epoch in range(num_epochs):
        epoch_start = time.time()
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels.max(1)[1]).sum().item()
        
        
        train_acc = 100. * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels.max(1)[1]).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        epoch_time = time.time() - epoch_start
        
        # 计算平均损失和准确率
        train_loss = train_loss/len(train_loader)
        train_acc = train_acc/len(train_loader)
        val_loss = val_loss/len(val_loader)
        val_acc = val_acc/len(val_loader)
        
        # 记录wandb信息
        wandb_log = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }
        # 记录训练信息到wandb
        wandb.log(wandb_log)
        
        train_info = f"""Epoch [{epoch+1}/{num_epochs}]
                       Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}%
                       Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}%
                       耗时: {epoch_time:.2f} s, 预计剩余时间: {epoch_time*(num_epochs-epoch-1)/60:.2f} min
                       --------------------"""
        
        # 打印训练信息
        print(train_info)

        
        
if __name__ == '__main__':
    model = VisionTransformer()
    
    # for file_name in os.listdir("./"):
    #     if file_name.endswith('.pth'):
    #         # 加载当前目录下的pth文件
    #         model.load_state_dict(torch.load(file_name))
    #         print(f"loading {file_name}")
    #         break
        
    train_model(model, train_loader, val_loader)
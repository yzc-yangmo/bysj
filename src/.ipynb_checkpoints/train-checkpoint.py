import os, json, time
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader 

from data.dataset import FoodImageDataset
from model import Model

# 读取配置文件
'''
dataset:
    name: 模型名称，vit or resnet
    num_classes: 类别数量
    drop_rate: 随机失活率
    train_path: 训练集路径
    val_path: 验证集路径
    batch_size: 批量大小
    lr: 学习率
    num_epochs: 训练轮数
    weight_decay: 权重衰减
    use_wandb: 是否使用wandb记录训练信息
    transform_type: 数据增强类型，0: 简单变换，1: 数据增强，2: 增强数据增强
    notes: 备注

inference:
    name: 模型名称，vit or resnet
    num_classes: 类别数量，总共101类
    drop_rate: 随机失活率
    model_path: 模型路径

'''

config = json.load(open('./config.json'))

# 检查配置文件是否正确
if str(config["train"]["num_classes"]) not in config["train"]["dataset"]["train_path"]:
    raise ValueError("配置文件错误，num_classes 与 train_path 不匹配")

demo_id = time.strftime('%Y%m%d%H%M%S')
demo_name = f"{config["train"]['name']}_{config['train']['batch_size']}_{config['train']['lr']}_{config["train"]['drop_rate']}_DA-{config['train']['transform_type']}"

# 打印超参数
print(f"----------------config----------------")
for k, v in config.items():
    print(k)
    for kk, vv in v.items():
        print(f"--{kk}: {vv}")

# 读取数据集，训练集使用数据增强
transform_type = config["train"]["transform_type"]
train_foodimages = FoodImageDataset(config["train"]["dataset"]["train_path"], transform_type=transform_type)  # 使用数据增强
val_foodimages = FoodImageDataset(config["train"]["dataset"]["val_path"], transform_type = 0)  # 验证集使用简单变换
train_loader = DataLoader(train_foodimages, batch_size=config["train"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_foodimages, batch_size=config["train"]["batch_size"], shuffle=True)

# 训练函数
def train_model(model, train_loader, val_loader):
    # 配置wandb

    if use_wandb:
        wandb.init(project = f"bysj-chn-food（num_classes = {config['train']['num_classes']}）", 
                name = demo_name,
                config = config)
        wandb_log = {}
    
    # 根据配置文件定义超参数
    lr = config["train"]["lr"]
    weight_decay = config["train"]["weight_decay"]
    num_epochs = config["train"]["num_epochs"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device}!")
    
    # 移动模型到指定设备
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    
    best_val_acc = 0.0
    
    print(f"trian start! \nDemo_Id: {demo_id} \nDemo_Name: {demo_name}")
    for epoch in range(num_epochs):
        epoch_start = time.time()
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader):
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
            if not os.path.exists('./pth'):
                os.makedirs('./pth')
            pth_path = f'./pth/{demo_name}-{demo_id}-best_model.pth'
            torch.save(model.state_dict(), pth_path)
            print(f"model saved successfully, path {pth_path}")
        
        epoch_time = time.time() - epoch_start
        
        # 计算平均损失
        train_loss, val_loss = train_loss/len(train_loader), val_loss/len(val_loader)
        
        
        # 记录wandb信息
        if use_wandb:
            wandb_log = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time" : epoch_time
            }
            # 记录训练信息到wandb
            wandb.log(wandb_log)
        
        # 打印训练信息
        print(f"{'='*50}\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}\nDemo Name: {demo_name}\n{'-'*50}\nEpoch [{epoch+1}/{num_epochs}]\nTrain Loss: {train_loss:.4f}      Val Loss: {val_loss:.4f}\nTrain Accuracy: {train_acc:.2f}%   Val Accuracy: {val_acc:.2f}%\nEpoch Time: {epoch_time:.2f} s \n")

if __name__ == '__main__':
    model = Model(config["train"]["name"]).get_model()
    
    use_wandb = config["train"]["use_wandb"]
    print("-----------------model----------------\n", model, "\n--------------------------------")
    model.load_state_dict(torch.load("vit_128_0.001_0.3_DA-2-20250328203947-best_model.pth", weights_only=True))
    
    train_model(model, train_loader, val_loader)

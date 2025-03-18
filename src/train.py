import os, json, time
import wandb
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader 


from data.dataset import FoodImageDataset
from models import vit, resnet


# 读取配置文件
config = json.load(open('./config.json'))
mapping = json.load(open('./mapping.json'))

# 检查配置文件是否正确
if config["model"]["num_classes"] == 101 and config["dataset"]["train_path"] != "../dataset/train":
    raise ValueError("配置文件错误，num_classes为 101 时，train_path必须为../dataset/train")

if str(config["model"]["num_classes"]) not in config["dataset"]["train_path"]:
    raise ValueError("配置文件错误，num_classes 与 train_path 不匹配")


# 配置wandb
'''
project: 项目名称
name & demo_name: 实验名称，格式：模型-batch_size-learning_rate-drop_rate
config: 参数配置
'''

demo_id = time.strftime('%Y%m%d%H%M%S')
demo_name = f"{config['model']['name']}-{config['train']['batch_size']}-{config['train']['lr']}-{config['model']['drop_rate']}"
wandb.init(project = f"sub-food-image-classification（num_classes = {config['model']['num_classes']}）", 
           name = demo_name,
           config = config)
wandb_log = {}

# 打印超参数
print(f"----------------config----------------")
for k, v in config.items():
    print(k)
    for kk, vv in v.items():
        print(f"--{kk}: {vv}")

# 读取数据集
train_foodimages = FoodImageDataset(config["dataset"]["train_path"])
val_foodimages = FoodImageDataset(config["dataset"]["val_path"])
train_loader = DataLoader(train_foodimages, batch_size=config["train"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_foodimages, batch_size=config["train"]["batch_size"], shuffle=True)

# 训练函数
def train_model(model, train_loader, val_loader):
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
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_acc = 0.0
    
    print(f"trian start! \nDemo_Id: {demo_id} \nDemo_Name: {demo_name}")
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
            torch.save(model.state_dict(), f'./pth/{demo_name}-{demo_id}-best_model.pth')
        
        epoch_time = time.time() - epoch_start
        
        # 计算平均损失
        train_loss, val_loss = train_loss/len(train_loader), val_loss/len(val_loader)
        
        # 记录wandb信息
        wandb_log = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]['lr']
        }
        # 记录训练信息到wandb
        wandb.log(wandb_log)
        
        # 打印训练信息
        print(f"{'='*50}\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}\nDemo Name: {demo_name}\n{'-'*50}\nEpoch [{epoch+1}/{num_epochs}]\nTrain Loss: {train_loss:.4f}      Val Loss: {val_loss:.4f}\nTrain Accuracy: {train_acc:.2f}%   Val Accuracy: {val_acc:.2f}%\nEpoch Time: {epoch_time:.2f} s \n")

        
if __name__ == '__main__':
    model = vit.VisionTransformer()
    print("-----------------model----------------\n", model, "\n--------------------------------")
    # for file_name in os.listdir("./"):
    #     if file_name.endswith('.pth'):
    #         # 加载当前目录下的pth文件
    #         model.load_state_dict(torch.load(file_name))
    #         print(f"loading {file_name}")
    #         break
        
    train_model(model, train_loader, val_loader)
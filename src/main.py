import os
import sys
from PyQt5.QtWidgets import QApplication
from gui import FoodRecognitionSystem

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 读取配置文件
'''
dataset:
    name: 模型名称，vit or resnet
    num_classes: 类别数量，总共101类
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

def main():
    app = QApplication(sys.argv)
    window = FoodRecognitionSystem()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
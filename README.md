# 食物识别系统

## 数据集

食物图像数据集使用 Hugging Face 上的公开数据集 [food-chinese-2017](https://huggingface.co/datasets/chaeso/food_chinese_2017)

共 256 个食物类别，每个类别大约 200 张图片，图像尺寸为 256x256。

## 模型

模型使用 Vision Transformer 模型。

## 训练

训练使用 128 的批量大小，使用 1000 的训练轮数，使用 0.01 的权重衰减，使用 5e-4 的学习率。

```bash
cd src
python/python3 train.py
```

## 推理

```bash
python main.py
```


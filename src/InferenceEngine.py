import os, json, time
import torch
from PIL import Image
from data.transform import FoodImageTransform
from model import Model

os.chdir(os.path.dirname(os.path.abspath(__file__)))

config = json.load(open("config.json", "r", encoding="utf-8"))
# {food_id: {chn: str, calories: int, protein: int, fat: int, carb: int}}
food_info = json.load(open("food-info-50.json", "r", encoding="utf-8"))

class InferenceEngine:
    def __init__(self, model_name=None):
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 定义图像预处理
        self.transform = FoodImageTransform(transform_type=0)
        # 加载模型
        if model_name is not None:
            self.model = self.load_model_by_name(model_name)
        else:   
            self.model = self.load_model()
    
    # 读取配置文件加载模型
    def load_model(self):
        try:
            # 加载模型
            model_name = config["inference"]["name"]
            model = Model(model_name).get_model()
            print(f"use : {model_name}")
            
            # 读取模型参数
            model_state_path = config["inference"]["model_path"]
            if not os.path.exists(model_state_path):
                raise FileNotFoundError(f"not found model state file: {model_state_path}")
            state_dict = torch.load(model_state_path, map_location=self.device)
            model.load_state_dict(state_dict)
            
            # 将模型移动到GPU并设置为评估模式
            model.to(self.device)
            model.eval()
            return model
        
        except Exception as e:
            raise Exception(f"load model failed: {e}")
    
    # 加载指定名称的模型
    def load_model_by_name(self, model_name):   
        try:
            # 加载模型
            model = Model(model_name).get_model()
            print(f"use : {model_name}")
            
            # 读取模型参数
            for i in os.listdir("./pth/best"):
                if model_name in i:
                    model_state_path = os.path.join("./pth/best", i)
                    break
                
            if not os.path.exists(model_state_path):
                raise FileNotFoundError(f"not found model state file: {model_state_path}")
            state_dict = torch.load(model_state_path, map_location=self.device)
            model.load_state_dict(state_dict)
            
            # 将模型移动到GPU并设置为评估模式
            model.to(self.device)
            model.eval()
            return model
        
        except Exception as e:
            raise Exception(f"load model failed: {e}") 
    
    
    def inference(self, image_path):
        try:
            start_time = time.time()
            # 加载图像
            image = Image.open(image_path).convert("RGB")
            # 图像预处理
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)# 添加批次维度并移动到GPU
            
            # 推理
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
            # 获取最可能类别的索引和置信度
            confidence, class_idx = torch.max(probabilities, 0)
            confidence_value = confidence.item() * 100
            class_idx = class_idx.item()
            food_name = food_info[str(class_idx)]["chn"]
            # 食物的营养信息
            nutrition_info = {
                "calories": food_info[str(class_idx)]["calories"],
                "protein": food_info[str(class_idx)]["protein"],
                "fat": food_info[str(class_idx)]["fat"],
                "carb": food_info[str(class_idx)]["carb"]
            }
            return {
                "success": True,
                "class_idx": class_idx, # int, 食物类别索引
                "food_name": food_name, # str, 食物名称
                "confidence": confidence_value, # float, 置信度
                "nutrition_info": nutrition_info, # dict, 食物营养信息
                "inference_time": time.time() - start_time # float, 推理时间
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
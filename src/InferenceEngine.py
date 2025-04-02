import os, json, time
import torch
from PIL import Image
from data.transform import FoodImageTransform
from model import Model

os.chdir(os.path.dirname(os.path.abspath(__file__)))

config = json.load(open("config.json", "r", encoding="utf-8"))
# {food_id: {chn: str, calories: int, protein: int, fat: int, carb: int}}
food_info = json.load(open("food-info.json", "r", encoding="utf-8"))

class InferenceEngine:
    def __init__(self, model_path):
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载模型
        self.model = self.load_model(model_path)
        # 定义图像预处理
        self.transform = FoodImageTransform(transform_type=0)
    

    def load_model(self, model_state_path):
        try:
            # 加载模型
            model = Model(config["inference"]["name"]).get_model()
            
            # 读取模型参数
            if not os.path.exists(model_state_path):
                raise FileNotFoundError(f"not found model state file: {model_state_path}")
            
            state_dict = torch.load(model_state_path, map_location=self.device)
            model.load_state_dict(state_dict)
            
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
                "food_name": food_name,
                "confidence": confidence_value,
                "nutrition_info": nutrition_info,
                "inference_time": time.time() - start_time
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
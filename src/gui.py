import os, json
import torch
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QScrollArea, QGridLayout, QGroupBox, QStatusBar)
from PyQt5.QtGui import QPixmap,QFont, QIcon
from PyQt5.QtCore import Qt
from InferenceEngine import InferenceEngine

os.chdir(os.path.dirname(os.path.abspath(__file__)))

config = json.load(open("config.json", "r", encoding="utf-8"))

class FoodRecognitionSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.load_model()
        self.first_upload = True # 第一次上传完成后，将first_upload设置为False，再次上传时无需点击即可进行识别

    # 创建推理实例
    def load_model(self):
        try:
            
            model_state_path = config["inference"]["model_path"]
            self.InferenceEngine_engine = InferenceEngine(model_state_path)
            print("model load success, model path: ", model_state_path)
        
        except Exception as e:
            self.statusBar.showMessage(f'发生错误: {str(e)}')
            self.result_label.setText(f"发生错误: {str(e)}")
    
    def initUI(self):
        
        # 设置窗口标题和大小
        device_info = "CPU"
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_names = []
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_names.append(f"GPU {i}: {gpu_name}")
            device_info = " | ".join(gpu_names)
        self.setWindowTitle(f'食物识别系统 ({device_info})')
        self.setGeometry(100, 100, 1000, 800)
        self.setWindowIcon(QIcon('./food101.ico')) 
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 标题
        title_label = QLabel('食物识别系统')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont('Arial', 20, QFont.Bold))
        main_layout.addWidget(title_label)
        
        # 内容区域
        content_layout = QHBoxLayout()
        
        # 左侧区域 - 图片显示
        self.image_display = QLabel('请上传食物图片')
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setMinimumSize(400, 400)
        self.image_display.setStyleSheet("border: 2px dashed #aaa; background-color: #f5f5f5;")
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.image_display)
        
        content_layout.addWidget(scroll_area, 3)
        
        # 右侧区域 - 识别结果和控制按钮
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 控制按钮组
        control_group = QGroupBox("操作")
        control_layout = QGridLayout()
        
        self.upload_btn = QPushButton('上传图片')
        self.upload_btn.clicked.connect(self.upload_image)
        self.recognize_btn = QPushButton('识别食物')
        self.recognize_btn.clicked.connect(self.recognize_food)
        self.clear_btn = QPushButton('清除')
        self.clear_btn.clicked.connect(self.clear_all)
        
        control_layout.addWidget(self.upload_btn, 0, 0)
        control_layout.addWidget(self.recognize_btn, 0, 1)
        control_layout.addWidget(self.clear_btn, 1, 0, 1, 2)
        
        control_group.setLayout(control_layout)
        right_layout.addWidget(control_group)
        
        # 识别结果组
        result_group = QGroupBox("识别结果")
        result_layout = QVBoxLayout()
        
        self.result_label = QLabel('尚未识别任何食物')
        self.result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.result_label.setWordWrap(True)
        self.result_label.setMinimumHeight(300)
        self.result_label.setStyleSheet("background-color: white; padding: 10px;")
        
        result_layout.addWidget(self.result_label)
        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)
        
        content_layout.addWidget(right_panel, 2)
        main_layout.addLayout(content_layout)
        
        # 状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('准备就绪')
        
    def upload_image(self):
        """上传图片功能"""
        file_path, _ = QFileDialog.getOpenFileName(self, '选择图片', '', 
                                                  'Images (*.png *.jpg *.jpeg)')
        if file_path:
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                # 调整图片大小以适应显示区域，保持纵横比
                pixmap = pixmap.scaled(self.image_display.width(), 
                                      self.image_display.height(),
                                      Qt.KeepAspectRatio, 
                                      Qt.SmoothTransformation)
                self.image_display.setPixmap(pixmap)
                self.statusBar.showMessage(f'已加载图片: {file_path}')
                self.current_image_path = file_path
                
                if not self.first_upload:
                    self.recognize_food()
                
            else:
                self.statusBar.showMessage('无法加载图片')
    
    def recognize_food(self):
        """识别食物功能"""
        if hasattr(self, 'current_image_path'):
            self.statusBar.showMessage('正在识别...')
            
            try:
                # 进行推理
                infer_info = self.InferenceEngine_engine.inference(self.current_image_path)
                
                if infer_info["success"]:
                    # 构建结果显示
                    result_text = f"""
                    <p><b>食物名称:</b> {infer_info["food_info"]["chn"]}</p>
                    <p><b>置信度:</b> {infer_info["confidence"]:.1f}%</p>
                    <p><b>耗时:</b> {infer_info["inference_time"]:.2f} 秒</p>
                    <p><br></p>
                    <p><b>营养成分:</b></p>
                    <p><b>热量:</b> {infer_info["food_info"]["calories"]} kcal</p>
                    <p><b>蛋白质:</b> {infer_info["food_info"]["protein"]} g</p>
                    <p><b>脂肪:</b> {infer_info["food_info"]["fat"]} g</p>
                    <p><b>碳水化合物:</b> {infer_info["food_info"]["carb"]} g </p>
                    <p><b>以上数据为100g食物的营养成分</b></p>
                    """
                    
                    self.result_label.setText(result_text)
                    self.statusBar.showMessage('识别完成')
                    
                    # 第一次上传完成后，将first_upload设置为False
                    if self.first_upload:
                        self.first_upload = False
                
                else:
                    self.statusBar.showMessage(f'识别失败: {infer_info["error"]}')
                    self.result_label.setText(f"识别失败: {infer_info['error']}")
            
            except Exception as e:
                self.statusBar.showMessage(f'发生错误: {str(e)}')
                self.result_label.setText(f"发生错误: {str(e)}")
        else:
            self.statusBar.showMessage('请先上传图片')
    
    def clear_all(self):
        """清除所有内容"""
        self.image_display.setText('请上传食物图片')
        self.image_display.setPixmap(QPixmap())  # 清除图片
        self.result_label.setText('尚未识别任何食物')
        if hasattr(self, 'current_image_path'):
            delattr(self, 'current_image_path')
        self.statusBar.showMessage('已清除所有内容')
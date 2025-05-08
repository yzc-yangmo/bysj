import os, json, time, random
import torch
import csv
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QScrollArea, QGridLayout, QGroupBox, QStatusBar)
from PyQt5.QtGui import QPixmap,QFont, QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from InferenceEngine import InferenceEngine

os.chdir(os.path.dirname(os.path.abspath(__file__)))

config = json.load(open("config.json", "r", encoding="utf-8"))

class FoodRecognitionSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.load_model()
        self.first_upload = True # 第一次上传完成后，将first_upload设置为False，再次上传时无需点击即可进行识别
        self.type = "single" # 识别类型，single: 单张图片识别，dir: 文件夹批量识别

    # 创建推理实例
    def load_model(self):
        try:
            self.InferenceEngine = InferenceEngine()
            print("model load success!")
        
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
        self.upload_dir_btn = QPushButton('上传文件夹/批量识别')
        self.upload_dir_btn.clicked.connect(self.upload_dir)
        self.recognize_btn = QPushButton('识别食物')
        self.recognize_btn.clicked.connect(self.recognize_food)
        self.clear_btn = QPushButton('清除')
        self.clear_btn.clicked.connect(self.clear_all)
        
        control_layout.addWidget(self.upload_btn, 0, 0)
        control_layout.addWidget(self.upload_dir_btn, 0, 1)
        control_layout.addWidget(self.recognize_btn, 1, 0)
        control_layout.addWidget(self.clear_btn, 1, 1)
        
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
    
    # 在界面中显示图片
    def show_image(self, image_path):
        """显示图片"""
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(self.image_display.width(), 
                              self.image_display.height(),
                              Qt.KeepAspectRatio, 
                              Qt.SmoothTransformation)
        self.image_display.setPixmap(pixmap)
    
    def upload_image(self):
        """上传图片功能"""
        file_path, _ = QFileDialog.getOpenFileName(self, '选择图片', '', 
                                                  'Images (*.png *.jpg *.jpeg)')
        if file_path:
            self.show_image(file_path)
            self.statusBar.showMessage(f'已加载图片: {file_path}')
            self.current_image_path = file_path
            self.type = "single" # 设置识别类型为单张图片识别
            
            # 如果不是第一次上传完成后，则直接进行识别
            if not self.first_upload:
                self.recognize_single()
                
        else:
            self.statusBar.showMessage('无法加载图片')

    def recognize_single(self):
        """识别食物功能"""
        if hasattr(self, 'current_image_path'):
            self.statusBar.showMessage('正在识别...')
            
            try:
                # 进行推理
                infer_info = self.InferenceEngine.inference(self.current_image_path)
                
                if infer_info["success"]:
                    # 构建结果显示
                    result_text = f"""
                    <p><b>食物名称:</b> {infer_info["food_name"]}</p>
                    <p><b>置信度:</b> {infer_info["confidence"]:.1f}%</p>
                    <p><b>耗时:</b> {infer_info["inference_time"]:.4f} 秒</p>
                    <p><br></p>
                    <p><b>营养成分:</b></p>
                    <p><b>热量:</b> {infer_info["nutrition_info"]["calories"]} kcal</p>
                    <p><b>蛋白质:</b> {infer_info["nutrition_info"]["protein"]} g</p>
                    <p><b>脂肪:</b> {infer_info["nutrition_info"]["fat"]} g</p>
                    <p><b>碳水化合物:</b> {infer_info["nutrition_info"]["carb"]} g </p>
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
    
    def upload_dir(self):
        """上传文件夹功能"""
        dir_path = QFileDialog.getExistingDirectory(self, '选择文件夹')
        if dir_path:
            self.current_dir_path = dir_path
            # 显示文件夹中的第一张图片
            first_image = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][0]
            self.show_image(os.path.join(dir_path, first_image)) 
            # 更新状态栏
            self.statusBar.showMessage(f'已选择文件夹: {dir_path}')
            # 设置识别类型为文件夹批量识别
            self.type = "dir" 
    
    def recognize_dir(self):
        """批量识别文件夹中的图片"""
        if not hasattr(self, 'current_dir_path'):
            self.statusBar.showMessage('请先选择文件夹')
            return

        # 支持的图片格式
        image_extensions = ('.jpg', '.jpeg', '.png')
        
        # 获取文件夹中所有图片文件
        image_files = [f for f in os.listdir(self.current_dir_path) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            self.statusBar.showMessage('文件夹中没有找到支持的图片文件')
            return

        # 准备CSV文件
        recognition_results_dir = "./recognition-results"
        csv_path = os.path.join(recognition_results_dir, f'recognition-results-{time.strftime("%Y%m%d%H%M%S")}.csv')
        csv_headers = ['图片名称', '食物名称', '置信度', '识别耗时(秒)', 
                      '热量(kcal)', '蛋白质(g)', '脂肪(g)', '碳水化合物(g)']
        if not os.path.exists(recognition_results_dir):
            os.makedirs(recognition_results_dir)
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(csv_headers)
                
                total_images = len(image_files)
                for i, image_file in enumerate(image_files, 1):
                    
                    image_path = os.path.join(self.current_dir_path, image_file)
                    self.statusBar.showMessage(f'processing: {image_file} ({i}/{total_images})')
                    
                    # 处理UI事件，使界面能够更新
                    QApplication.processEvents()
                    
                    # 在界面中显示图片
                    self.show_image(image_path)
                    
                    # 进行推理
                    infer_info = self.InferenceEngine.inference(image_path)
                    
                    if infer_info["success"]:
                        writer.writerow([
                            image_file,
                            infer_info["food_name"],
                            f"{infer_info['confidence']:.1f}%",
                            f"{infer_info['inference_time']:.2f}",
                            infer_info["nutrition_info"]["calories"],
                            infer_info["nutrition_info"]["protein"],
                            infer_info["nutrition_info"]["fat"],
                            infer_info["nutrition_info"]["carb"]
                        ])
                        # 更新识别结果
                        result_text = f"""
                        <p><b>进度:</b> {i}/{total_images}</p>
                        <p><b>图片名称:</b> {image_file}</p>
                        <p><b>食物名称:</b> {infer_info["food_name"]}</p>
                        <p><b>置信度:</b> {infer_info["confidence"]:.1f}%</p>
                        <p><b>耗时:</b> {infer_info["inference_time"]:.4f} 秒</p>
                        <p><br></p>
                        <p><b>营养成分:</b></p>
                        <p><b>热量:</b> {infer_info["nutrition_info"]["calories"]} kcal</p>
                        <p><b>蛋白质:</b> {infer_info["nutrition_info"]["protein"]} g</p>
                        <p><b>脂肪:</b> {infer_info["nutrition_info"]["fat"]} g</p>
                        <p><b>碳水化合物:</b> {infer_info["nutrition_info"]["carb"]} g </p>
                        <p><b>以上数据为100g食物的营养成分</b></p>
                        """
                        self.result_label.setText(result_text)
                    else:
                        writer.writerow([
                            image_file,
                            f"识别失败: {infer_info['error']}",
                            "", "", "", "", "", ""
                        ])
                    
                    # 再次处理UI事件，确保每次推理后界面都能更新
                    QApplication.processEvents()

            self.statusBar.showMessage(f'批量识别完成，结果已保存至: {os.path.abspath(csv_path)}')
            self.result_label.setText(f'批量识别完成!\n共处理 {total_images} 张图片\n耗时: {time.time() - start_time:.2f} 秒\n结果已保存至: {os.path. abspath(csv_path)}')
            
        except Exception as e:
            self.statusBar.showMessage(f'批量识别过程中发生错误: {str(e)}')
            self.result_label.setText(f"批量识别过程中发生错误: {str(e)}")
    
    # 识别食物功能，根据识别类型选择识别方式
    def recognize_food(self):
        """识别食物功能"""
        if self.type == "single":
            self.recognize_single()
        elif self.type == "dir":
            self.recognize_dir()
    
    def clear_all(self):
        """清除所有内容"""
        self.image_display.setText('请上传食物图片')
        self.image_display.setPixmap(QPixmap())  # 清除图片
        self.result_label.setText('尚未识别任何食物')
        if hasattr(self, 'current_image_path'):
            delattr(self, 'current_image_path')
        self.statusBar.showMessage('已清除所有内容')
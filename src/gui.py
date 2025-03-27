import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QScrollArea, QGridLayout, QGroupBox, QStatusBar)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QSize

class FoodRecognitionSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle('食物识别系统')
        self.setGeometry(100, 100, 1000, 800)
        
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
            else:
                self.statusBar.showMessage('无法加载图片')
    
    def recognize_food(self):
        """识别食物功能"""
        if hasattr(self, 'current_image_path'):
            # 这里应该调用实际的食物识别模型
            # 以下是模拟的识别结果
            self.statusBar.showMessage('正在识别...')
            
            # 模拟识别结果 - 实际应用中应替换为真实的识别逻辑
            result_text = """
            <h3>识别结果:</h3>
            <p><b>食物名称:</b> 披萨</p>
            <p><b>置信度:</b> 95.7%</p>
            <p><b>营养成分:</b></p>
            <ul>
                <li>热量: 266千卡/100克</li>
                <li>蛋白质: 11克/100克</li>
                <li>脂肪: 10克/100克</li>
                <li>碳水化合物: 33克/100克</li>
            </ul>
            """
            
            self.result_label.setText(result_text)
            self.statusBar.showMessage('识别完成')
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

def main():
    app = QApplication(sys.argv)
    window = FoodRecognitionSystem()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

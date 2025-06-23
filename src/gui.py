import sys
import os
import random
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QFrame
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt

# --- 从 main.py 迁移过来的核心代码 ---

# 1. 路径定义
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

PROTOTXT_PATH = os.path.join(MODEL_DIR, 'sphereface_deploy.prototxt')
CAFFEMODEL_PATH = os.path.join(MODEL_DIR, 'sphereface_model.caffemodel')

AR_DATABASE_DIR = os.path.join(DATA_DIR, '数据库-AR_k120_s26_w80_h100', 'AR_k120_s26_w80_h100')

DB_FEATURE_PATH = os.path.join(BASE_DIR, 'feature_db.npy')
DB_LABELS_PATH = os.path.join(BASE_DIR, 'db_labels.npy')

# 2. 核心功能函数
def get_face_feature(image, net):
    """提取单张人脸图像的特征(无直方图均衡化)"""
    resized_image = cv2.resize(image, (96, 112))
    normalized_image = (resized_image.astype(np.float32) - 127.5) / 128.0
    blob = cv2.dnn.blobFromImage(normalized_image)
    net.setInput(blob)
    try:
        feature = net.forward('fc5')
        return feature.flatten()
    except Exception as e:
        print(f"[错误] 特征提取失败: {e}")
        return None

def recognize_face(probe_feature, database, db_labels):
    """在数据库中识别单张人脸"""
    if probe_feature is None or database.shape[0] == 0:
        return -1, -1.0
    
    # L2归一化
    probe_feature /= np.linalg.norm(probe_feature)
    database /= np.linalg.norm(database, axis=1, keepdims=True)
    
    similarities = np.dot(database, probe_feature)
    
    best_match_index = np.argmax(similarities)
    best_score = similarities[best_match_index]
    predicted_label = db_labels[best_match_index]
    
    return predicted_label, best_score

# --- PyQt5 GUI 应用 ---

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸识别系统")
        self.setGeometry(100, 100, 800, 600)
        
        self.net = None
        self.feature_db = None
        self.db_labels = None
        self.test_images = []
        self.test_labels = []
        self.gallery_images = {} # 存储每个ID的一张代表性注册照片
        self.current_test_image = None
        self.current_true_label = None

        self.load_all_data()
        self.init_ui()

    def load_all_data(self):
        print("1. 正在加载模型和数据库...")
        # 加载模型
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
        # 加载特征数据库
        self.feature_db = np.load(DB_FEATURE_PATH)
        self.db_labels = np.load(DB_LABELS_PATH)
        print("   - 模型和数据库加载成功。")

        print("2. 正在加载图像数据...")
        # 加载所有AR图片用于显示
        images, labels = [], []
        for filename in sorted(os.listdir(AR_DATABASE_DIR)):
             if filename.endswith('.tif'):
                filepath = os.path.join(AR_DATABASE_DIR, filename)
                label = int(filename[2:5])
                image = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                images.append(image)
                labels.append(label)
        
        # 划分数据集，获取测试集和注册集代表图像
        count = {}
        for img, label in zip(images, labels):
            count.setdefault(label, 0)
            if count[label] < 7: # 前7张是注册
                if label not in self.gallery_images:
                    self.gallery_images[label] = img
            else: # 之后是测试
                self.test_images.append(img)
                self.test_labels.append(label)
            count[label] += 1
        print(f"   - 图像加载完毕: {len(self.test_images)}张测试图片, {len(self.gallery_images)}张注册代表图片。")


    def init_ui(self):
        # 主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 标题
        title_label = QLabel("人脸识别演示")
        title_label.setFont(QFont('Arial', 24))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 图片显示区
        image_layout = QHBoxLayout()
        main_layout.addLayout(image_layout)
        
        # 左侧：待识别区
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.create_styled_label("待识别图片 (测试集)", 16))
        self.test_image_label = self.create_image_label()
        left_layout.addWidget(self.test_image_label)
        self.true_label_display = self.create_styled_label("真实ID: ?", 14)
        left_layout.addWidget(self.true_label_display)
        
        # 右侧：识别结果区
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.create_styled_label("识别结果 (注册库)", 16))
        self.result_image_label = self.create_image_label()
        right_layout.addWidget(self.result_image_label)
        self.predicted_label_display = self.create_styled_label("预测ID: ?", 14)
        right_layout.addWidget(self.predicted_label_display)
        
        image_layout.addLayout(left_layout)
        image_layout.addStretch()
        image_layout.addLayout(right_layout)
        
        # 按钮区
        button_layout = QHBoxLayout()
        self.pick_button = QPushButton("随机选择图片")
        self.pick_button.clicked.connect(self.pick_random_image)
        self.recognize_button = QPushButton("开始识别")
        self.recognize_button.clicked.connect(self.run_recognition)
        
        button_layout.addStretch()
        button_layout.addWidget(self.pick_button)
        button_layout.addWidget(self.recognize_button)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)
        
        # 状态/结果显示
        self.final_result_label = QLabel("请先选择一张图片")
        self.final_result_label.setFont(QFont('Arial', 18))
        self.final_result_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.final_result_label)
        main_layout.addStretch()

    def create_image_label(self):
        label = QLabel("请选择图片")
        label.setFixedSize(240, 300) # 3 * 80, 3 * 100
        label.setFrameShape(QFrame.Box)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        return label
        
    def create_styled_label(self, text, size):
        label = QLabel(text)
        label.setFont(QFont('Arial', size))
        label.setAlignment(Qt.AlignCenter)
        return label

    def pick_random_image(self):
        # 随机选择一张测试图片
        random_index = random.randint(0, len(self.test_images) - 1)
        self.current_test_image = self.test_images[random_index]
        self.current_true_label = self.test_labels[random_index]
        
        # 显示图片
        self.display_image(self.current_test_image, self.test_image_label)
        self.true_label_display.setText(f"真实ID: {self.current_true_label}")
        
        # 重置结果区
        self.result_image_label.clear()
        self.result_image_label.setText("?")
        self.predicted_label_display.setText("预测ID: ?")
        self.final_result_label.setText('已选择图片，请点击"开始识别"')
        self.final_result_label.setStyleSheet("")

    def run_recognition(self):
        if self.current_test_image is None:
            self.final_result_label.setText("请先随机选择一张图片！")
            return
            
        # 提取特征并识别
        probe_feature = get_face_feature(self.current_test_image, self.net)
        predicted_label, score = recognize_face(probe_feature, self.feature_db, self.db_labels)
        
        # 显示结果
        result_image = self.gallery_images.get(predicted_label, self.create_image_label()) # 如果找不到就显示空
        self.display_image(result_image, self.result_image_label)
        self.predicted_label_display.setText(f"预测ID: {predicted_label} (置信度: {score:.2f})")
        
        # 判断对错并更新状态
        if predicted_label == self.current_true_label:
            self.final_result_label.setText("识别正确！")
            self.final_result_label.setStyleSheet("color: green;")
        else:
            self.final_result_label.setText("识别错误！")
            self.final_result_label.setStyleSheet("color: red;")
            
    def display_image(self, cv_img, label_widget):
        """将OpenCV图像转换为QPixmap并显示在QLabel上"""
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_img)
        label_widget.setPixmap(pixmap.scaled(label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = FaceRecognitionApp()
    main_win.show()
    sys.exit(app.exec_()) 
import os
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# --- 1. 路径和核心函数定义 (大部分与main.py相同) ---

# 路径定义
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

PROTOTXT_PATH = os.path.join(MODEL_DIR, 'sphereface_deploy.prototxt')
CAFFEMODEL_PATH = os.path.join(MODEL_DIR, 'sphereface_model.caffemodel')

# VIDEO_FRAMES_DIR = os.path.join(DATA_DIR, '人脸视频（图像序列）数据库', 'db')
VIDEO_FRAMES_DIR = os.path.join(DATA_DIR, '真实采集的人脸数据')

# 核心函数
def get_face_feature(image, net):
    """提取单张人脸图像的特征"""
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

def build_feature_database(images, labels, net):
    """构建特征数据库，为每个人计算平均特征向量"""
    features_map = defaultdict(list)
    print("正在为注册库（Gallery）构建特征...")
    for img, label in tqdm(zip(images, labels), total=len(images), desc="构建数据库"):
        feature = get_face_feature(img, net)
        if feature is not None:
            features_map[label].append(feature)
            
    database, db_labels = [], []
    print("正在计算平均特征向量...")
    for label, features in tqdm(sorted(features_map.items())):
        if features:
            avg_feature = np.mean(features, axis=0)
            database.append(avg_feature)
            db_labels.append(label)
            
    return np.array(database), np.array(db_labels)

def recognize_face(probe_feature, database, db_labels):
    """在数据库中识别单张人脸"""
    if probe_feature is None or database.shape[0] == 0:
        return -1, -1.0
    
    probe_feature /= np.linalg.norm(probe_feature)
    database /= np.linalg.norm(database, axis=1, keepdims=True)
    similarities = np.dot(database, probe_feature)
    
    best_match_index = np.argmax(similarities)
    best_score = similarities[best_match_index]
    predicted_label = db_labels[best_match_index]
    
    return predicted_label, best_score

# --- 2. 针对视频帧数据集的新数据加载器 ---

def load_video_frame_dataset(dataset_path):
    """
    加载视频帧数据集。
    该数据集已按 train/test 文件夹组织好。
    """
    train_images, train_labels = [], []
    test_images, test_labels = [], []
    
    print(f"开始从 {dataset_path} 加载视频帧数据集...")
    if not os.path.exists(dataset_path):
        print(f"[错误] 数据集路径不存在: {dataset_path}")
        return ([], []), ([], [])

    subject_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    for subject_folder in tqdm(subject_folders, desc="加载对象"):
        try:
            label = int(subject_folder.split('_')[0])
        except (ValueError, IndexError):
            print(f"[警告] 无法从文件夹名解析标签: {subject_folder}")
            continue
            
        # 加载训练图片
        train_path = os.path.join(dataset_path, subject_folder, 'train')
        if os.path.exists(train_path):
            for filename in os.listdir(train_path):
                if filename.lower().endswith('.jpg'):
                    filepath = os.path.join(train_path, filename)
                    image = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is not None:
                        train_images.append(image)
                        train_labels.append(label)

        # 加载测试图片
        test_path = os.path.join(dataset_path, subject_folder, 'test')
        if os.path.exists(test_path):
            for filename in os.listdir(test_path):
                if filename.lower().endswith('.jpg'):
                    filepath = os.path.join(test_path, filename)
                    image = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is not None:
                        test_images.append(image)
                        test_labels.append(label)
                        
    print(f"加载完毕: {len(train_images)}张训练图片, {len(test_images)}张测试图片, 共{len(subject_folders)}个类别。")
    return (train_images, np.array(train_labels)), (test_images, np.array(test_labels))


# --- 3. 主执行流程 ---

if __name__ == '__main__':
    # 加载模型
    print("正在加载SphereFace模型...")
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
    print("模型加载成功。")

    # 加载视频帧数据集
    (train_images, train_labels), (test_images, test_labels) = load_video_frame_dataset(VIDEO_FRAMES_DIR)
    
    if not train_images or not test_images:
        print("数据加载失败，程序退出。")
    else:
        # 构建特征数据库
        feature_db, db_labels = build_feature_database(train_images, train_labels, net)
        print(f"特征数据库构建完毕，包含 {len(db_labels)} 个人的特征。")

        # 在测试集上评估
        correct_predictions = 0
        print("正在视频帧测试集上进行评估...")
        for i in tqdm(range(len(test_images)), desc="评估进度"):
            probe_image = test_images[i]
            true_label = test_labels[i]

            probe_feature = get_face_feature(probe_image, net)
            predicted_label, score = recognize_face(probe_feature, feature_db, db_labels)

            if predicted_label == true_label:
                correct_predictions += 1
        
        # 计算并打印准确率
        accuracy = (correct_predictions / len(test_images)) * 100 if test_images else 0
        print(f"\n评估完成！")
        print(f"总测试帧数: {len(test_images)}")
        print(f"正确识别帧数: {correct_predictions}")
        print(f"识别准确率: {accuracy:.2f}%") 
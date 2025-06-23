import os
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# --- 1. 路径和核心函数定义 (与之前脚本相同) ---

# 路径定义
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

PROTOTXT_PATH = os.path.join(MODEL_DIR, 'sphereface_deploy.prototxt')
CAFFEMODEL_PATH = os.path.join(MODEL_DIR, 'sphereface_model.caffemodel')

FERET_DATABASE_DIR = os.path.join(DATA_DIR, '数据库-feret_k175_s7_w80_h80', 'feret_k175_s7_w80_h80')

# 核心函数
def get_face_feature(image, net):
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
    if probe_feature is None or database.shape[0] == 0:
        return -1, -1.0
    
    probe_feature /= np.linalg.norm(probe_feature)
    database /= np.linalg.norm(database, axis=1, keepdims=True)
    similarities = np.dot(database, probe_feature)
    
    best_match_index = np.argmax(similarities)
    best_score = similarities[best_match_index]
    predicted_label = db_labels[best_match_index]
    
    return predicted_label, best_score

# --- 2. 针对Feret数据集的数据加载和划分函数 ---

def load_feret_dataset(dataset_path):
    """
    加载Feret人脸数据集，并根据文件名中的ID和序号排序。
    """
    image_tuples = []
    print(f"开始从 {dataset_path} 加载Feret数据集...")
    if not os.path.exists(dataset_path):
        print(f"[错误] 数据集路径不存在: {dataset_path}")
        return [], []

    for filename in os.listdir(dataset_path):
        if filename.lower().endswith('.bmp'):
            try:
                label = int(filename[:3])
                img_index = int(filename[4:6])
                
                filepath = os.path.join(dataset_path, filename)
                image = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if image is not None:
                    image_tuples.append((image, label, img_index))
                else:
                    print(f"[警告] 无法读取图像: {filepath}")
            except (ValueError, IndexError):
                print(f"[警告] 无法解析文件名: {filename}")

    # 根据人物ID和图片序号排序，确保数据一致性
    image_tuples.sort(key=lambda x: (x[1], x[2]))
    
    images = [t[0] for t in image_tuples]
    labels = [t[1] for t in image_tuples]
    
    print(f"成功加载 {len(images)} 张图像，共 {len(set(labels))} 个类别。")
    return images, np.array(labels)

def split_data_per_person(images, labels, n_train_per_person=3):
    """
    按人划分数据集，每人取n张作为训练集，其余作为测试集。
    """
    train_images, train_labels = [], []
    test_images, test_labels = [], []
    
    count = defaultdict(int)
    
    for img, label in zip(images, labels):
        if count[label] < n_train_per_person:
            train_images.append(img)
            train_labels.append(label)
        else:
            test_images.append(img)
            test_labels.append(label)
        count[label] += 1
        
    return (train_images, np.array(train_labels)), (test_images, np.array(test_labels))


# --- 3. 主执行流程 ---

if __name__ == '__main__':
    # 加载模型
    print("正在加载SphereFace模型...")
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
    print("模型加载成功。")

    # 加载并划分Feret数据集
    all_images, all_labels = load_feret_dataset(FERET_DATABASE_DIR)
    (train_images, train_labels), (test_images, test_labels) = split_data_per_person(all_images, all_labels, n_train_per_person=3)
    
    if not train_images or not test_images:
        print("数据加载或划分失败，程序退出。")
    else:
        print(f"数据集划分完毕: {len(train_images)} 张训练图片, {len(test_images)} 张测试图片。")
        # 构建特征数据库
        feature_db, db_labels = build_feature_database(train_images, train_labels, net)
        print(f"特征数据库构建完毕，包含 {len(db_labels)} 个人的特征。")

        # 在测试集上评估
        correct_predictions = 0
        print("正在Feret测试集上进行评估...")
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
        print(f"总测试样本数: {len(test_images)}")
        print(f"正确识别数: {correct_predictions}")
        print(f"识别准确率: {accuracy:.2f}%")

        # 贝叶斯分类器对比实验
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score
        print("\n[对比实验] 使用高斯贝叶斯分类器进行识别...")

        # 1. 提取所有训练图片的特征
        train_features = []
        for img in train_images:
            feat = get_face_feature(img, net)
            train_features.append(feat)
        train_features = np.array(train_features)

        # 2. 提取所有测试图片的特征
        test_features = []
        for img in test_images:
            feat = get_face_feature(img, net)
            test_features.append(feat)
        test_features = np.array(test_features)

        # 3. 训练贝叶斯分类器
        clf = GaussianNB()
        clf.fit(train_features, train_labels)

        # 4. 预测
        pred_labels = clf.predict(test_features)

        # 5. 计算准确率
        acc = accuracy_score(test_labels, pred_labels)
        print(f"贝叶斯分类器识别准确率: {acc*100:.2f}%")# 贝叶斯分类器对比实验
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score
        print("\n[对比实验] 使用高斯贝叶斯分类器进行识别...")

        # 1. 提取所有训练图片的特征
        train_features = []
        for img in train_images:
            feat = get_face_feature(img, net)
            train_features.append(feat)
        train_features = np.array(train_features)

        # 2. 提取所有测试图片的特征
        test_features = []
        for img in test_images:
            feat = get_face_feature(img, net)
            test_features.append(feat)
        test_features = np.array(test_features)

        # 3. 训练贝叶斯分类器
        clf = GaussianNB()
        clf.fit(train_features, train_labels)

        # 4. 预测
        pred_labels = clf.predict(test_features)

        # 5. 计算准确率
        acc = accuracy_score(test_labels, pred_labels)
        print(f"贝叶斯分类器识别准确率: {acc*100:.2f}%") 
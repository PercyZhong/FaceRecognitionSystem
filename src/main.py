import os
import cv2
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 定义项目相关的路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# SphereFace 模型文件路径
PROTOTXT_PATH = os.path.join(MODEL_DIR, 'sphereface_deploy.prototxt')
CAFFEMODEL_PATH = os.path.join(MODEL_DIR, 'sphereface_model.caffemodel')

# AR 数据集路径
AR_DATABASE_DIR = os.path.join(DATA_DIR, '数据库-AR_k120_s26_w80_h100', 'AR_k120_s26_w80_h100')

def load_ar_dataset(dataset_path):
    """
    加载AR人脸数据集
    :param dataset_path: AR数据集的路径
    :return: (images, labels)元组，images是图像列表，labels是对应的标签列表
    """
    images = []
    labels = []
    print(f"开始从 {dataset_path} 加载AR数据集...")

    if not os.path.exists(dataset_path):
        print(f"[错误] 数据集路径不存在: {dataset_path}")
        return [], []
    for filename in os.listdir(dataset_path):
        if filename.endswith('.tif'):
            try:
                # 从文件名中提取标签 (例如 'AR001-1.tif' -> 1)
                label = int(filename[2:5])            
                filepath = os.path.join(dataset_path, filename)   
                # 读取图像，使用imdecode处理中文路径问题
                image = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)        
                if image is not None:
                    # 将单通道图像转换为3通道BGR图像，以匹配模型输入
                    if len(image.shape) == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    images.append(image)
                    labels.append(label)
                else:
                    print(f"[警告] 无法读取图像: {filepath}")
            except (ValueError, IndexError) as e:
                print(f"[警告] 无法解析文件名: {filename} - {e}")  
    print(f"成功加载 {len(images)} 张图像，共 {len(set(labels))} 个类别。")
    return images, np.array(labels)

def get_face_feature(image, net, use_hist_eq=True):
    """
    使用SphereFace模型提取单张人脸图像的特征
    :param image: 输入的人脸图像 (BGR格式)
    :param net: 加载好的Caffe模型
    :param use_hist_eq: 是否对图像应用CLAHE（对比度受限的自适应直方图均衡化）
    :return: 512维的特征向量，如果提取失败则返回None
    """
    # 1. 图像预处理：可选的CLAHE
    if use_hist_eq:
        # 将BGR图像转换到YCrCb颜色空间，只对亮度(Y)通道进行处理
        img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = img_ycrcb[:, :, 0]
        
        # 创建CLAHE对象 (Contrast Limited Adaptive Histogram Equalization)
        # 这种方法在小区域内进行直方图均衡化，可以更好地保留细节并避免过度增强对比度。
        # 调整tileGridSize以适应图像尺寸，避免产生过多块状伪影
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        
        # 将CLAHE应用于Y通道
        clahe_y_channel = clahe.apply(y_channel)
        
        # 将处理后的Y通道与原始Cr、Cb通道合并
        img_ycrcb[:, :, 0] = clahe_y_channel
        
        # 将处理后的图像转换回BGR颜色空间
        processed_image = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        processed_image = image

    # 2. 图像后续处理
    # 调整图像尺寸至 96x112
    resized_image = cv2.resize(processed_image, (96, 112))
    
    # 归一化：(pixel - 127.5) / 128
    normalized_image = (resized_image.astype(np.float32) - 127.5) / 128.0
    
    # 3. 准备输入Blob
    blob = cv2.dnn.blobFromImage(normalized_image)
    
    # 4. 将Blob输入网络
    net.setInput(blob)
    
    # 5. 前向传播，得到特征
    try:
        # 'fc5'是SphereFace模型输出特征向量的层名
        feature = net.forward('fc5')
        return feature.flatten()
    except Exception as e:
        print(f"[错误] 特征提取失败: {e}")
        return None

def split_data_per_person(images, labels, n_train_per_person=7):
    """
    按人划分数据集，每人取n张作为训练集，其余作为测试集
    :param images: 图像列表
    :param labels: 标签列表
    :param n_train_per_person: 每人用于训练的样本数
    :return: (train_images, train_labels), (test_images, test_labels)
    """
    train_images, train_labels = [], []
    test_images, test_labels = [], []
    
    count = defaultdict(int)
    
    # 将图像和标签打包在一起处理，以保持对应关系
    for img, label in zip(images, labels):
        if count[label] < n_train_per_person:
            train_images.append(img)
            train_labels.append(label)
        else:
            test_images.append(img)
            test_labels.append(label)
        count[label] += 1
        
    return (train_images, np.array(train_labels)), (test_images, np.array(test_labels))

def build_feature_database(images, labels, net, use_hist_eq=True):
    """
    构建特征数据库，为每个人计算平均特征向量
    :param images: 训练图像列表
    :param labels: 训练标签列表
    :param net: 加载好的Caffe模型
    :param use_hist_eq: 是否对图像应用CLAHE（对比度受限的自适应直方图均衡化）
    :return: (database, db_labels)元组，database是平均特征向量列表，db_labels是对应的标签
    """
    # 1. 为所有训练图片提取特征
    features_map = defaultdict(list)
    print("正在为注册库（Gallery）构建特征...")
    for img, label in tqdm(zip(images, labels), total=len(images)):
        feature = get_face_feature(img, net, use_hist_eq)
        if feature is not None:
            features_map[label].append(feature)
            
    # 2. 计算每个人的平均特征
    database = []
    db_labels = []
    print("正在计算平均特征向量...")
    for label, features in tqdm(sorted(features_map.items())):
        if features:
            avg_feature = np.mean(features, axis=0)
            database.append(avg_feature)
            db_labels.append(label)
            
    return np.array(database), np.array(db_labels)

def recognize_face(probe_feature, database, db_labels):
    """
    在数据库中识别单张人脸
    :param probe_feature: 待识别图像的特征向量
    :param database: 特征数据库 (N, 512)
    :param db_labels: 数据库中每个特征对应的标签
    :return: (predicted_label, best_score) 元组
    """
    if probe_feature is None or database.shape[0] == 0:
        return -1, -1.0
        
    # 计算余弦相似度
    # probe_feature需要reshape成 (1, 512) 来和 (N, 512) 的database计算
    similarities = cosine_similarity(probe_feature.reshape(1, -1), database)
    
    # 找到分数最高的位置
    best_match_index = np.argmax(similarities)
    best_score = similarities[0, best_match_index]
    predicted_label = db_labels[best_match_index]
    
    return predicted_label, best_score

if __name__ == '__main__':
    USE_HIST_EQ = False  #  开关：设置为True来启用CLAHE

    # 1. 加载Caffe模型
    print("正在加载SphereFace模型...")
    if os.path.exists(PROTOTXT_PATH) and os.path.exists(CAFFEMODEL_PATH):
        net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
        print("模型加载成功。")
    else:
        print("[错误] 模型文件不存在，请检查路径。")
        net = None

    if net:
        # 2. 加载并划分AR数据集
        all_images, all_labels = load_ar_dataset(AR_DATABASE_DIR)
        (train_images, train_labels), (test_images, test_labels) = split_data_per_person(all_images, all_labels, n_train_per_person=7)
        print(f"数据集划分完毕: {len(train_images)} 张训练图片, {len(test_images)} 张测试图片。")

        # 3. 构建特征数据库
        feature_db, db_labels = build_feature_database(train_images, train_labels, net, use_hist_eq=USE_HIST_EQ)
        print(f"特征数据库构建完毕，包含 {len(db_labels)} 个人的特征。")

        # 4. 在测试集上进行评估
        correct_predictions = 0
        print("正在测试集上进行评估...")
        for i in tqdm(range(len(test_images))):
            probe_image = test_images[i]
            true_label = test_labels[i]

            # 提取测试图片的特征
            probe_feature = get_face_feature(probe_image, net, use_hist_eq=USE_HIST_EQ)
            
            # 进行识别
            predicted_label, score = recognize_face(probe_feature, feature_db, db_labels)

            if predicted_label == true_label:
                correct_predictions += 1
        
        # 5. 计算并打印准确率
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
            feat = get_face_feature(img, net, use_hist_eq=USE_HIST_EQ)
            train_features.append(feat)
        train_features = np.array(train_features)

        # 2. 提取所有测试图片的特征
        test_features = []
        for img in test_images:
            feat = get_face_feature(img, net, use_hist_eq=USE_HIST_EQ)
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

        # 6. 将特征数据库保存到文件，以供GUI使用
        db_filepath = os.path.join(BASE_DIR, 'feature_db.npy')
        labels_filepath = os.path.join(BASE_DIR, 'db_labels.npy')
        np.save(db_filepath, feature_db)
        np.save(labels_filepath, db_labels)
        print(f"\n特征数据库已保存到 {db_filepath}")
        print(f"数据库标签已保存到 {labels_filepath}") 
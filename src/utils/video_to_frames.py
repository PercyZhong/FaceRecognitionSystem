import cv2
import os
import argparse

def extract_frames(video_path, output_folder, file_prefix):
    """
    从视频中提取人脸帧并保存为图片。

    :param video_path: 输入的视频文件路径。
    :param output_folder: 保存帧的输出文件夹。
    :param file_prefix: 保存的图片文件名前缀。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载Haar人脸检测器
    face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 将帧转换为灰度图像以进行人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        # 只处理检测到的最大的人脸
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True) # 按面积排序
            x, y, w, h = faces[0]
            
            # 裁剪出人脸区域
            face_img = frame[y:y+h, x:x+w]
            
            # 保存帧，使用imencode来处理包含非ASCII字符的路径
            output_filename = f"{file_prefix}_{count:04d}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            try:
                is_success, im_buf_arr = cv2.imencode(".jpg", face_img)
                if is_success:
                    im_buf_arr.tofile(output_path)
                    count += 1
            except Exception as e:
                print(f"写入文件时出错: {output_path} - {e}")
            
    cap.release()
    print(f"处理完成！从 {video_path} 提取了 {count} 个人脸帧到 {output_folder}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从视频中提取人脸帧")
    parser.add_argument("--video", required=True, help="输入的视频文件路径")
    parser.add_argument("--output", required=True, help="保存人脸帧的输出文件夹")
    parser.add_argument("--prefix", default="frame", help="保存的图片文件名前缀")

    args = parser.parse_args()

    extract_frames(args.video, args.output, args.prefix)

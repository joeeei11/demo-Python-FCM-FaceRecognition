### src/data_loader.py
import os
import cv2
import numpy as np


class DataLoader:
    def __init__(self, data_dir='dataset', target_size=(64, 64)):
        """
        初始化数据加载器
        :param data_dir: 数据集根目录
        :param target_size: 统一图像大小 (宽, 高)
        """
        self.data_dir = data_dir
        self.target_size = target_size
        self.label_map = {}  # 名字到整数ID的映射
        self.inv_label_map = {}  # 整数ID到名字的映射

    def cv2_imread_safe(self, file_path):
        """
        [关键修复] 能够读取中文路径的 OpenCV 读取函数
        """
        try:
            # 1. 使用 numpy 读取二进制流
            img_array = np.fromfile(file_path, np.uint8)
            # 2. 解码为图像
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            # 打印错误但不中断程序，方便排查
            print(f"[Warning] 读取失败: {file_path}, 错误: {e}")
            return None

    def load_data(self):
        """
        遍历目录加载图像，转换为灰度并展平。
        :return: X (样本数, 特征数), y (标签列表)
        """
        images = []
        labels = []

        if not os.path.exists(self.data_dir):
            print(f"[Warning] 数据目录 '{self.data_dir}' 不存在。请先运行 tools/prepare_data.py 生成数据。")
            return np.array([]), np.array([])

        # 获取所有子文件夹（即人名），排序保证 ID 顺序一致
        person_names = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]

        # 建立标签映射
        for idx, name in enumerate(sorted(person_names)):
            self.label_map[name] = idx
            self.inv_label_map[idx] = name

        print(f"[Info] 检测到 {len(person_names)} 个类别: {person_names}")

        valid_count = 0
        for name in person_names:
            person_dir = os.path.join(self.data_dir, name)
            file_list = os.listdir(person_dir)

            for file_name in file_list:
                file_path = os.path.join(person_dir, file_name)

                # [Fix] 使用支持中文路径的安全读取方法
                img = self.cv2_imread_safe(file_path)

                if img is None:
                    continue


                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


                resized = cv2.resize(gray, self.target_size)


                equalized = cv2.equalizeHist(resized)


                flat_img = equalized.flatten()

                images.append(flat_img)
                labels.append(self.label_map[name])
                valid_count += 1

        X = np.array(images)
        y = np.array(labels)

        print(f"[Info] 数据加载完成: {valid_count} 张图像，特征维度 {X.shape[1] if X.size > 0 else 0}")
        return X, y

    def get_label_name(self, label_idx):
        """根据ID获取人名"""
        return self.inv_label_map.get(label_idx, "Unknown")
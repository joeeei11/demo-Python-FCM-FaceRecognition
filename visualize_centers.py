import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
# 必须导入原来的类，否则 pickle 无法加载
from fcm_model import FaceFCM

warnings.filterwarnings("ignore")


def visualize_cluster_centers(model_path='face_system.pkl', image_size=(64, 64)):
    print(f"正在加载模型: {model_path} ...")

    try:
        # 1. 加载模型
        fcm = FaceFCM.load_model(model_path)
    except FileNotFoundError:
        print("错误：未找到模型文件。请先运行 main.py 进行训练。")
        return

    # 获取聚类中心 (shape: n_clusters x n_components)
    centers_features = fcm.cntr

    # 获取对应的标签名称
    cluster_labels = fcm.cluster_label_map
    label_names = fcm.label_names

    print("正在逆变换特征...")

    # 2. 逆标准化 (Inverse Scaling)
    # 也就是反向执行 (x * scale) + mean
    centers_pca = fcm.scaler.inverse_transform(centers_features)

    # 3. 逆 PCA (Inverse PCA)
    # 将 40 维向量还原回 4096 维像素向量
    centers_flat = fcm.pca.inverse_transform(centers_pca)

    # 4. 绘图
    n_clusters = fcm.n_clusters
    cols = 5
    rows = (n_clusters // cols) + (1 if n_clusters % cols > 0 else 0)

    plt.figure(figsize=(15, 3 * rows))
    plt.suptitle('FCM Cluster Centers ("Average Faces")', fontsize=16)

    for i in range(n_clusters):
        ax = plt.subplot(rows, cols, i + 1)

        # Reshape 回图像尺寸 (64, 64)
        face_img = centers_flat[i].reshape(image_size)

        # 获取该簇代表的人名
        label_id = cluster_labels.get(i, -1)
        person_name = label_names.get(label_id, "Unknown")

        plt.imshow(face_img, cmap='gray')
        plt.title(f"Cluster {i}\n({person_name})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    print("可视化完成。")


if __name__ == "__main__":
    # 确保您的字体支持（可选）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    visualize_cluster_centers()
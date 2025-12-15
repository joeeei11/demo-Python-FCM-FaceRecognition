### src/main.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")  # 屏蔽无关警告

from data_loader import DataLoader
from fcm_model import FaceFCM


def main():
    DATA_DIR = 'dataset'

    # 1. 加载数据
    loader = DataLoader(data_dir=DATA_DIR, target_size=(64, 64))
    X, y = loader.load_data()

    if len(X) == 0:
        print("[Error] 数据加载失败。请确保已运行 tools/prepare_data.py")
        return

    # 2. 划分数据
    # stratify=y 确保训练/测试集类别比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_classes = len(np.unique(y))
    print(f"总类别数: {num_classes}")
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 3. 初始化模型
    # Cluster 数量 = 类别数量
    # PCA 维度 = 40
    fcm = FaceFCM(n_clusters=num_classes, n_components=40, m=2.0)

    # 4. 训练
    fcm.train(X_train, y_train)
    fcm.set_label_names(loader.inv_label_map)
    fcm.save_model('face_system.pkl')

    # 5. 评估
    print("\n--- 评估报告 ---")
    y_pred = fcm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"总体准确率: {acc * 100:.2f}%")

    # [修复点] 显式指定我们要统计的 labels 列表
    # 这样即使 y_pred 包含 -1，也不会导致维度不匹配报错
    # -1 会被自动视为分类错误，降低 recall，但不影响程序运行
    valid_labels = sorted(np.unique(y))
    target_names = [loader.get_label_name(i) for i in valid_labels]

    try:
        print(classification_report(
            y_test,
            y_pred,
            labels=valid_labels,  # 强制指定只关注有效ID (0-18)
            target_names=target_names,
            zero_division=0  # 防止除以0警告
        ))
    except Exception as e:
        print(f"[Warning] 生成详细报告时出错: {e}")

    # 6. 演示
    print("\n[随机测试]")
    # 防止测试集过小导致报错
    sample_count = min(3, len(X_test))
    if sample_count > 0:
        indices = np.random.choice(len(X_test), sample_count, replace=False)
        for idx in indices:
            sample = X_test[idx].reshape(1, -1)
            true_name = loader.get_label_name(y_test[idx])
            pred_name = fcm.predict_name(sample)[0]

            # 如果预测出 -1，pred_name 会是 Unknown
            res = "SUCCESS" if true_name == pred_name else "FAIL"
            print(f"真: {true_name:<6} | 预: {pred_name:<6} -> {res}")


if __name__ == "__main__":
    main()
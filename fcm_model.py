### src/fcm_model.py
import numpy as np
import skfuzzy as fuzz
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import pickle


class FaceFCM:
    def __init__(self, n_clusters=19, n_components=40, m=2.0, error=0.005, max_iter=1000):
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.m = m
        self.error = error
        self.max_iter = max_iter


        self.pca = PCA(n_components=n_components, whiten=True)
        self.scaler = StandardScaler()

        self.cntr = None
        self.cluster_label_map = {}
        self.label_names = {}

    def set_label_names(self, label_map):
        self.label_names = label_map

    def _transform(self, X, fit=False):
        """内部特征转换管道"""
        if fit:
            X_pca = self.pca.fit_transform(X)
            X_scaled = self.scaler.fit_transform(X_pca)
        else:
            X_pca = self.pca.transform(X)
            X_scaled = self.scaler.transform(X_pca)
        return X_scaled

    def train(self, X, y):
        print("[Step 1] 特征工程 (PCA + Scaling)...")
        # 1. 提取特征
        X_features = self._transform(X, fit=True)
        print(f"   -> 特征矩阵形状: {X_features.shape}")

        print(f"[Step 2] FCM 聚类 (Clusters={self.n_clusters})...")
        # 2. 训练 FCM
        # 转置为 (n_features, n_samples)
        cntr, u, _, _, _, _, fpc = fuzz.cmeans(
            data=X_features.T,
            c=self.n_clusters,
            m=self.m,
            error=self.error,
            maxiter=self.max_iter,
            init=None,
            seed=42
        )
        self.cntr = cntr
        print(f"   -> FPC (聚类清晰度): {fpc:.4f}")

        print("[Step 3] 建立 聚类中心 -> 标签 映射...")
        # 3. 标签映射 (Hard Assignment)
        # 统计每个 Cluster 里哪个 Label 最多
        cluster_votes = defaultdict(list)

        # 获取每个样本最大隶属度的 Cluster ID
        hard_clusters = np.argmax(u, axis=0)

        for i, cluster_id in enumerate(hard_clusters):
            true_label = y[i]
            cluster_votes[cluster_id].append(true_label)

        self.cluster_label_map = {}
        active_count = 0
        for c in range(self.n_clusters):
            labels = cluster_votes[c]
            if labels:
                # 选出现次数最多的标签
                most_common = max(set(labels), key=labels.count)
                self.cluster_label_map[c] = most_common
                active_count += 1
            else:
                self.cluster_label_map[c] = -1  # 空簇

        print(f"   -> 有效聚类覆盖率: {active_count}/{self.n_clusters}")

    def predict(self, X_new):
        if self.cntr is None: raise Exception("模型未训练")

        # 1. 特征转换
        X_features = self._transform(X_new, fit=False)

        # 2. 预测隶属度
        u_predicted, _, _, _, _, _ = fuzz.cmeans_predict(
            X_features.T, self.cntr, self.m, self.error, self.max_iter
        )

        # 3. 取最大隶属度对应的 Cluster
        cluster_indices = np.argmax(u_predicted, axis=0)

        # 4. 查表得到 Label
        predicted_labels = [self.cluster_label_map.get(c, -1) for c in cluster_indices]

        return np.array(predicted_labels)

    def predict_name(self, X_new):
        ids = self.predict(X_new)
        return [self.label_names.get(i, "Unknown") for i in ids]

    def save_model(self, filepath):
        with open(filepath, 'wb') as f: pickle.dump(self, f)

    @classmethod
    def load_model(cls, filepath):
        with open(filepath, 'rb') as f: return pickle.load(f)
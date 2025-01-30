import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch.nn.functional as F
import torch

def resize_feature_map_torch(feature_map, target_size=(7, 7)):
    """
    将特征图插值为固定大小（PyTorch实现）。
    Args:
        feature_map (torch.Tensor): 输入特征图，形状为 (C, H, W)。
        target_size (tuple): 目标大小 (H', W')。
    Returns:
        torch.Tensor: 调整后的特征图，形状为 (C, H', W')。
    """
    feature_map = torch.tensor(feature_map).unsqueeze(0)  # 添加 batch 维度 -> (1, C, H, W)
    resized_feature = F.interpolate(feature_map, size=target_size, mode='bilinear', align_corners=False)
    return resized_feature.squeeze(0).numpy()  # 去掉 batch 维度 -> (C, H', W')

def global_average_pooling(feature_map):
    """
    对特征图进行全局平均池化，将形状从 (C, H, W) 转为 (C,)。
    Args:
        feature_map (np.array): 输入特征图，形状为 (C, H, W)。
    Returns:
        np.array: 池化后的特征向量，形状为 (C,)。
    """
    return np.mean(feature_map, axis=(1, 2))  # 对 H 和 W 进行平均


class FeatureVisualizer:
    def __init__(self, output_dir="./visualization_results", max_samples=200):
        """
        初始化累积绘图工具

        Args:
            output_dir (str): 保存可视化结果的路径。
            max_samples (int): 累积到多少样本后停止并可视化。
        """
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.features = []  # 用于存储累积的特征
        self.sample_count = 0  # 当前累积样本数量
        os.makedirs(self.output_dir, exist_ok=True)

    def add_features(self, new_features):
        # 如果特征是 3 维 (C, H, W)，插值到固定大小
        if len(new_features.shape) == 3:
            new_features = resize_feature_map_torch(new_features, target_size=(224,224))  # 将特征统一为 (C, 7, 7)
        # 累积特征逻辑
        if len(self.features) == 0:
            self.features.append(new_features)
        elif new_features.shape == self.features[0].shape:
            self.features.append(new_features)
        else:
            raise ValueError(f"Feature dimensions mismatch: {new_features.shape} vs {self.features[0].shape}")

        self.sample_count += 1

        if self.sample_count >= self.max_samples:
            self.visualize_and_reset()


    def visualize_and_reset(self):
        """
        可视化当前累积的特征，并重置累积池。
        """
        print(f"Visualizing {self.sample_count} samples...")
        
        # 将所有累积的特征拼接起来
        processed_features = []
        for feature in self.features:
            # 使用全局平均池化或展平特征
            processed_feature = global_average_pooling(feature)  # 或者 flatten_feature_map(feature)
            processed_features.append(processed_feature)

        features_array = np.vstack(processed_features)  # 转换为 [N, D]

        # 动态调整 perplexity
        perplexity = min(30, max(5, self.sample_count - 1))  # perplexity 取样本数减 1 或最大 30，最小为 5
        print(f"Using perplexity={perplexity} for t-SNE.")

        # 使用 t-SNE 降维
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced_features = tsne.fit_transform(features_array)

        # 绘制可视化图
        plt.figure(figsize=(8, 8))
        plt.scatter(
            reduced_features[:, 0],
            reduced_features[:, 1],
            s=5,
            alpha=0.7,
            label=f"Samples: {self.sample_count}",
        )
        plt.title("t-SNE Visualization of Features")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend()
        plt.grid(True)

        # 保存结果
        save_path = os.path.join(self.output_dir, f"features_{self.sample_count}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Visualization saved to {save_path}")

        # 重置累积池
        self.features = []
        self.sample_count = 0



# # 示例使用：
# # 假设 `inputs[0]['image']` 是提取的特征
# visualizer = FeatureVisualizer(output_dir="./visualization_results", max_samples=128)

# # 模拟数据流：一次处理 10 个样本
# for i in range(20):  # 模拟 20 批次输入
#     batch_features = np.random.rand(10, 512)  # 替换为真实的特征
#     visualizer.add_features(batch_features)

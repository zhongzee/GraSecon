import os
import json
import numpy as np
from PIL import Image
from collections import defaultdict
import torch

class TextMeanFeatureCalculator:
    def __init__(self, save_dir, max_samples=None, device="cuda"):
        """
        初始化 TextMeanFeatureCalculator。
        
        Args:
            save_dir (str): 保存均值特征的目录。
            max_samples (int): 每个领域最多处理的特征数量。
            device (str): 设备类型，默认 "cuda"。
        """
        self.save_dir = save_dir
        self.max_samples = max_samples
        self.device = device
        self.features_by_domain = defaultdict(list)
        self.logger = logging.getLogger("TextMeanFeatureCalculator")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)

    def compute_mean_features(self, dataset_name, text_features):
        """
        计算文本特征的领域均值特征。
        
        Args:
            dataset_name (str): 数据集名称，用于区分保存路径。
            text_features (dict): 文本特征，键为领域名称，值为特征张量列表。
        
        Returns:
            mean_features (dict): 每个领域的均值特征，键为领域名称，值为均值特征张量。
        """
        dataset_save_dir = os.path.join(self.save_dir, dataset_name)
        os.makedirs(dataset_save_dir, exist_ok=True)
        
        mean_feature_paths = {domain: os.path.join(dataset_save_dir, f"{domain}_mean_features.npy") 
                              for domain in text_features.keys()}
        mean_features = {}

        # 检查是否已有保存的均值特征
        if all(os.path.exists(path) for path in mean_feature_paths.values()):
            for domain, path in mean_feature_paths.items():
                mean_features[domain] = torch.from_numpy(np.load(path)).to(self.device)
            self.logger.info(f"已加载数据集 {dataset_name} 的所有领域均值特征")
            return mean_features

        # 计算新的均值特征
        for domain, features in text_features.items():
            if len(features) > 0:
                stacked_features = torch.stack(features)  # 形状为 [N, C]
                mean_feature = stacked_features.mean(dim=0)  # 计算均值特征
                mean_features[domain] = mean_feature
                np.save(mean_feature_paths[domain], mean_feature.cpu().numpy())  # 保存到文件
                self.logger.info(f"已保存 {domain} 的均值特征到 {mean_feature_paths[domain]}")

        return mean_features
    

    def correct_domain_bias(self, text_features, domain_features, mean_domain_features):
        """
        校正文本特征的领域偏置。
        
        Args:
            text_features (torch.Tensor): 文本特征，形状为 [N, C]。
            domain_features (torch.Tensor): 领域特定的特征均值，形状为 [C]。
            mean_domain_features (torch.Tensor): 域不变的均值特征，形状为 [C]。
        
        Returns:
            corrected_text_features (torch.Tensor): 校正后的文本特征，形状与 text_features 相同。
        """
        t_hat = domain_features - mean_domain_features  # 计算领域偏置向量
        centered_text = text_features - t_hat  # 移除领域偏置
        norm = torch.norm(centered_text, p=2, dim=1, keepdim=True)  # L2 范数归一化
        corrected_text_features = centered_text / (norm + 1e-6)  # 避免除零

        return corrected_text_features

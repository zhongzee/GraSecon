import os
import json
import numpy as np
from PIL import Image
from collections import defaultdict
import torch
import logging


class TextMeanFeatureCalculator2:
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

    def compute_mean_features_gt(self, dataset_name, text_features):
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
        # if all(os.path.exists(path) for path in mean_feature_paths.values()):
        #     for domain, path in mean_feature_paths.items():
        #         mean_features[domain] = torch.from_numpy(np.load(path)).to(self.device)
        #     self.logger.info(f"已加载数据集 {dataset_name} 的所有领域均值特征")
        #     return mean_features

        # 计算新的均值特征
        for domain, features in text_features.items():
            if len(features) > 0:
                # 确保每个 feature 是 [C] 的形状
                features = [f.squeeze(0) if f.dim() == 2 and f.size(0) == 1 else f for f in features]

                # 堆叠特征
                stacked_features = torch.stack(features)  # [N, C]
                
                # 检查堆叠后的维度是否正确
                if stacked_features.dim() != 2:
                    raise ValueError(f"stacked_features has invalid shape: {stacked_features.shape}")

                # 计算均值特征
                mean_feature = stacked_features.mean(dim=0)  # [C]
                mean_features[domain] = mean_feature
                
                # 保存到文件
                np.save(mean_feature_paths[domain], mean_feature.cpu().numpy())  # 保存到文件
                self.logger.info(f"已保存 {domain} 的均值特征到 {mean_feature_paths[domain]}")


        return mean_features
    

    def compute_mean_features(self, dataset_name, text_features, sentence_type="detail", max_length=20):
        """
        计算文本特征的领域均值特征。

        Args:
            dataset_name (str): 数据集名称，用于区分保存路径。
            text_features (dict): 文本特征，键为领域名称，值为特征张量列表。
            sentence_type (str): 使用的句子类型，可选 "by_level", "candidate", "detail", "combined"。
            max_length (int): 对特征进行统一的最大长度（仅在特定场景下生效）。

        Returns:
            mean_features (dict): 每个领域的均值特征，键为领域名称，值为均值特征张量。
        """
        dataset_save_dir = os.path.join(self.save_dir, dataset_name)
        os.makedirs(dataset_save_dir, exist_ok=True)
        
        mean_feature_paths = {domain: os.path.join(dataset_save_dir, f"{domain}_mean_features.npy") 
                            for domain in text_features.keys()}
        mean_features = {}

        # # # 检查是否已有保存的均值特征
        # if all(os.path.exists(path) for path in mean_feature_paths.values()):
        #     for domain, path in mean_feature_paths.items():
        #         mean_features[domain] = torch.from_numpy(np.load(path)).to(self.device)
        #     self.logger.info(f"已加载数据集 {dataset_name} 的所有领域均值特征")
        #     return mean_features

        # 计算新的均值特征
        for domain, features in text_features.items():
            if len(features) > 0:
                if sentence_type in ["combined", "by_level","detail","candidate"]:
                    # 使用逐步计算均值的方式 inat
                    sum_feature = torch.zeros_like(features[0][0])  # 初始化累积张量
                    count = 0
                    for feature in features:
                        sum_feature += feature.mean(dim=0)  # 累积平均特征
                        count += 1
                    mean_feature = sum_feature / count  # 计算均值
                # # fsod
                # if sentence_type in ["combined", "by_level", "detail", "candidate"]:
                #     # 使用逐步计算均值的方式
                #     sum_feature = torch.zeros_like(features[0]) # 初始化累积张量并移动到设备
                #     count = 0
                #     for feature in features:
                #         # 确保 feature 是至少二维的
                #         if feature.dim() < 2:
                #             feature = feature.unsqueeze(0)  # 增加一个维度
                #         # 计算特征的均值并去除多余维度
                #         mean_feat = feature.mean(dim=0).squeeze()
                #         assert mean_feat.dim() == 1, f"Mean feature shape is not 1D: {mean_feat.shape}"
                #         sum_feature += mean_feat  # 累积平均特征
                #         count += 1
                #     mean_feature = sum_feature / count  # 计算均值
                #     assert mean_feature.dim() == 1, f"Final mean_feature shape is not 1D: {mean_feature.shape}"
                else:
                    # 使用 stack 的方式
                    # 确保每个 feature 是 [C] 的形状
                    features = [f.squeeze(0) if f.dim() == 2 and f.size(0) == 1 else f for f in features]
                    
                    # 堆叠特征
                    stacked_features = torch.stack(features)  # [N, C]
                    
                    # 检查堆叠后的维度是否正确
                    if stacked_features.dim() != 2:
                        raise ValueError(f"stacked_features has invalid shape: {stacked_features.shape}")
                    
                    # 计算均值特征
                    mean_feature = stacked_features.mean(dim=0)  # [C]

                # 保存均值特征
                mean_features[domain] = mean_feature
                np.save(mean_feature_paths[domain], mean_feature.cpu().numpy())  # 保存到文件
                self.logger.info(f"已保存 {domain} 的均值特征到 {mean_feature_paths[domain]}")

        return mean_features
    

    import os
    import torch
    import numpy as np

    def compute_mean_features_2(self, dataset_name, text_features, sentence_type="detail", max_length=20):
        """
        计算文本特征的领域均值特征。(适配orin偏置校准的范式，返回维度是C，但是仍然是累加)

        Args:
            dataset_name (str): 数据集名称，用于区分保存路径。
            text_features (dict): 文本特征，键为领域名称，值为特征张量列表。
            sentence_type (str): 使用的句子类型，可选 "by_level", "candidate", "detail", "combined"。
            max_length (int): 对特征进行统一的最大长度（仅在特定场景下生效）。

        Returns:
            mean_features (dict): 每个领域的均值特征，键为领域名称，值为均值特征张量（维度为 [C]）。
        """
        dataset_save_dir = os.path.join(self.save_dir, dataset_name)
        os.makedirs(dataset_save_dir, exist_ok=True)
        
        mean_feature_paths = {
            domain: os.path.join(dataset_save_dir, f"{domain}_mean_features.npy") 
            for domain in text_features.keys()
        }
        mean_features = {}

        # # # 检查是否已有保存的均值特征
        # if all(os.path.exists(path) for path in mean_feature_paths.values()):
        #     for domain, path in mean_feature_paths.items():
        #         mean_features[domain] = torch.from_numpy(np.load(path)).to(self.device)
        #     self.logger.info(f"已加载数据集 {dataset_name} 的所有领域均值特征")
        #     return mean_features

        # 计算新的均值特征
        for domain, features in text_features.items():
            if len(features) > 0:
                if sentence_type in ["combined", "by_level", "detail", "candidate"]:
                    # 使用逐步计算均值的方式
                    # 确保 features 是一个包含张量的列表
                    if not isinstance(features, list):
                        raise TypeError(f"Expected features for domain '{domain}' to be a list, but got {type(features)}")
                    
                    # 初始化 sum_feature
                    first_feature = features[0]
                    if not torch.is_tensor(first_feature):
                        raise TypeError(f"Expected features to contain torch.Tensor, but got {type(first_feature)}")
                    
                    sum_feature = torch.zeros_like(first_feature)  # 确保 sum_feature 与 features[0] 形状相同
                    print(f"Initialized sum_feature for domain '{domain}' with shape: {sum_feature.shape}")
                    
                    count = 0
                    for idx, feature in enumerate(features):
                        print(f"Domain: {domain}, Feature {idx} shape before processing: {feature.shape}")
                        
                        # 确保 feature 是至少二维的
                        if feature.dim() < 2:
                            feature = feature.unsqueeze(0)  # 增加一个维度
                            print(f"Domain: {domain}, Feature {idx} shape after unsqueeze: {feature.shape}")
                        
                        # 计算特征的均值，不使用 squeeze()
                        mean_feat = feature.mean(dim=0)
                        
                        # 检查 mean_feat 的维度
                        if mean_feat.dim() != 1:
                            raise ValueError(f"Mean feature shape is not 1D: {mean_feat.shape} for domain {domain}, feature index {idx}")
                        
                        print(f"Domain: {domain}, Feature {idx} mean feature shape: {mean_feat.shape}")
                        
                        sum_feature += mean_feat  # 累积平均特征
                        count += 1
                    
                    if count == 0:
                        print(f"Warning: No valid features found for domain '{domain}'. Skipping.")
                        continue

                    mean_feature = sum_feature / count  # 计算均值
                    
                    # 最终均值特征的维度检查
                    if mean_feature.dim() != 1:
                        raise ValueError(f"Final mean_feature shape is not 1D: {mean_feature.shape} for domain {domain}")
                else:
                    # 使用 stack 的方式，确保返回维度为 [C]
                    all_features = []
                    for feature in features:
                        if isinstance(feature, list):
                            all_features.extend(feature)
                        else:
                            all_features.append(feature)
                    
                    if len(all_features) == 0:
                        print(f"Warning: 领域 '{domain}' 中没有找到任何特征，跳过均值计算")
                        continue

                    # 堆叠特征并计算均值
                    stacked_features = torch.stack(all_features)  # [N, C]
                    mean_feature = stacked_features.mean(dim=0)  # [C]
                    
                    # 检查堆叠后的维度是否正确
                    if mean_feature.dim() != 1:
                        raise ValueError(f"mean_feature has invalid shape: {mean_feature.shape} for domain {domain}")
                
                # 保存均值特征
                mean_features[domain] = mean_feature
                np.save(mean_feature_paths[domain], mean_feature.cpu().numpy())  # 保存到文件
                self.logger.info(f"已保存 {domain} 的均值特征到 {mean_feature_paths[domain]}")

        return mean_features



    
    # def compute_mean_features(self, dataset_name, text_features, sentence_type="detail", max_length=20):
    #     """
    #     计算文本特征的领域均值特征。

    #     Args:
    #         dataset_name (str): 数据集名称，用于区分保存路径。
    #         text_features (dict): 文本特征，键为领域名称，值为特征张量列表。
    #         sentence_type (str): 使用的句子类型，可选 "by_level", "candidate", "detail", "combined"。
    #         max_length (int): 对特征进行统一的最大长度（仅在特定场景下生效）。

    #     Returns:
    #         mean_features (dict): 每个领域的均值特征，键为领域名称，值为均值特征张量。
    #     """
    #     dataset_save_dir = os.path.join(self.save_dir, dataset_name)
    #     os.makedirs(dataset_save_dir, exist_ok=True)
        
    #     mean_feature_paths = {domain: os.path.join(dataset_save_dir, f"{domain}_mean_features.npy") 
    #                         for domain in text_features.keys()}
    #     mean_features = {}

    #     # 检查是否已有保存的均值特征
    #     if all(os.path.exists(path) for path in mean_feature_paths.values()):
    #         for domain, path in mean_feature_paths.items():
    #             mean_features[domain] = torch.from_numpy(np.load(path)).to(self.device)
    #         self.logger.info(f"已加载数据集 {dataset_name} 的所有领域均值特征")
    #         return mean_features

    #     for domain, features in text_features.items():
    #         if len(features) > 0:
    #             if sentence_type in ["combined", "by_level",'candidate']:
    #                 print("使用逐步累积")
    #                 # 检查每个特征形状是否一致
    #                 feature_dim = features[0].shape[-1]
    #                 for feature in features:
    #                     if feature.shape[-1] != feature_dim:
    #                         raise ValueError(
    #                             f"Inconsistent feature dimensions in domain {domain}: "
    #                             f"expected {feature_dim}, but got {feature.shape[-1]}"
    #                         )
                    
    #                 # 使用逐步计算均值的方式
    #                 sum_feature = torch.zeros_like(features[0][0])  # 初始化累积张量
    #                 count = 0
    #                 for feature in features:
    #                     # 逐步累积特征的均值
    #                     sum_feature += feature.mean(dim=0)
    #                     count += 1
    #                 mean_feature = sum_feature / count  # 计算均值
    #             else:
    #                 # 使用 stack 的方式
    #                 # 确保每个 feature 是 [C] 的形状
    #                 features = [f.squeeze(0) if f.dim() == 2 and f.size(0) == 1 else f for f in features]

    #                 # 堆叠特征
    #                 stacked_features = torch.stack(features)  # [N, C]

    #                 # 检查堆叠后的维度是否正确
    #                 if stacked_features.dim() != 2:
    #                     raise ValueError(f"stacked_features has invalid shape: {stacked_features.shape}")
                    
    #                 # 计算均值特征
    #                 mean_feature = stacked_features.mean(dim=0)  # [C]

    #             # 保存均值特征
    #             mean_features[domain] = mean_feature
    #             np.save(mean_feature_paths[domain], mean_feature.cpu().numpy())  # 保存到文件
    #             self.logger.info(f"已保存 {domain} 的均值特征到 {mean_feature_paths[domain]}")


    #     return mean_features
    

    def correct_domain_bias(self, text_features, domain_features, cross_level_mean=None, global_mean=None):
        """
        校正文本特征的领域偏置。
        
        Args:
            text_features (torch.Tensor): 文本特征，形状为 [C] 或 [N, C]。
            domain_features (torch.Tensor): 当前粒度的领域均值特征，形状为 [C]。
            cross_level_mean (torch.Tensor, optional): 跨粒度层级的总体均值特征，形状为 [C]。
            global_mean (torch.Tensor, optional): 所有领域的总体均值特征，形状为 [C]。
        
        Returns:
            corrected_text_features (torch.Tensor): 校正后的文本特征，形状与 text_features 相同。
        """
        # 检查输入的形状是否符合预期
        if text_features.dim() == 1:  # 如果是 [C]
            text_features = text_features.unsqueeze(0)  # 转换为 [1, C]
        elif text_features.dim() != 2:
            raise ValueError(f"text_features shape is invalid: {text_features.shape}, expected [C] or [N, C]")
        
        if domain_features.dim() != 1:
            raise ValueError(f"domain_features shape is invalid: {domain_features.shape}, expected [C]")

        # 计算领域偏置向量
        t_hat = domain_features
        if cross_level_mean is not None:
            t_hat -= cross_level_mean  # 减去跨粒度层级均值
        if global_mean is not None:
            t_hat -= global_mean  # 减去所有领域的总体均值

        # 确保 t_hat 的形状可以广播到 text_features
        t_hat = t_hat.view(1, -1)  # [1, C]

        # 从文本特征中移除领域偏置
        centered_text = text_features - t_hat  # [N, C]

        # 计算 L2 范数
        norm = torch.norm(centered_text, p=2, dim=1, keepdim=True)  # [N, 1]

        # 避免除零并归一化
        corrected_text_features = centered_text / (norm + 1e-6)  # [N, C]

        # 如果原始输入是 [C]，返回时再转回 [C]
        if corrected_text_features.shape[0] == 1:
            return corrected_text_features.squeeze(0)
        return corrected_text_features
    
    
    # def correct_domain_bias_iNat(self,text_features, domain_features, cross_level_mean=None, global_mean=None):
    #     """
    #     校正文本特征的领域偏置。
        
    #     Args:
    #         text_features (torch.Tensor): 文本特征，形状为 [C] 或 [N, C]。
    #         domain_features (torch.Tensor): 当前粒度的领域均值特征，形状为 [C]。
    #         cross_level_mean (torch.Tensor, optional): 跨粒度层级的总体均值特征，形状为 [C]。
    #         global_mean (torch.Tensor, optional): 所有领域的总体均值特征，形状为 [C]。
    #         use_global_mean (bool): 是否使用 global_mean 进行偏置校正。
        
    #     Returns:
    #         corrected_text_features (torch.Tensor): 校正后的文本特征，形状与 text_features 相同。
    #     """
    #     # 检查输入的形状是否符合预期
    #     if text_features.dim() == 1:  # 如果是 [C]
    #         text_features = text_features.unsqueeze(0)  # 转换为 [1, C]
    #     elif text_features.dim() != 2:
    #         raise ValueError(f"text_features shape is invalid: {text_features.shape}, expected [C] or [N, C]")
        
    #     if domain_features.dim() != 1:
    #         raise ValueError(f"domain_features shape is invalid: {domain_features.shape}, expected [C]")

    #     # 计算领域偏置向量
    #     t_hat = domain_features.clone()
    #     if cross_level_mean is not None:
    #         t_hat -= cross_level_mean  # 减去跨粒度层级均值
    #     if global_mean is not None:
    #         t_hat -= global_mean  # 减去所有领域的总体均值

    #     # 确保 t_hat 的形状可以广播到 text_features
    #     t_hat = t_hat.view(1, -1)  # [1, C]

    #     # 从文本特征中移除领域偏置
    #     centered_text = text_features - t_hat  # [N, C]

    #     # 计算 L2 范数
    #     norm = torch.norm(centered_text, p=2, dim=1, keepdim=True)  # [N, 1]

    #     # 避免除零并归一化
    #     corrected_text_features = centered_text / (norm + 1e-6)  # [N, C]

    #     # 如果原始输入是 [C]，返回时再转回 [C]
    #     if corrected_text_features.shape[0] == 1:
    #         return corrected_text_features.squeeze(0)
    #     return corrected_text_features

    def correct_domain_bias_iNat(self, text_features, domain_features, cross_level_mean=None, global_mean=None, policy='no_gm'):
        """
        校正文本特征的领域偏置。
        
        Args:
            text_features (torch.Tensor): 文本特征，形状为 [C] 或 [N, C]。
            domain_features (torch.Tensor): 当前粒度的领域均值特征，形状为 [C]。
            cross_level_mean (torch.Tensor, optional): 跨粒度层级的总体均值特征，形状为 [C]。
            global_mean (torch.Tensor, optional): 所有领域的总体均值特征，形状为 [C]。
            policy (str): 决定是否使用 global_mean 的策略，值为 'gm' 或 'no_gm'。
        
        Returns:
            corrected_text_features (torch.Tensor): 校正后的文本特征，形状与 text_features 相同。
        """
        # 参数验证
        if not isinstance(policy, str):
            raise TypeError(f"policy should be a str, but got {type(policy)}")
        if policy not in ['gm', 'no_gm']:
            raise ValueError(f"policy should be 'gm' or 'no_gm', but got {policy}")
        
        # 检查输入的形状是否符合预期
        if text_features.dim() == 1:  # 如果是 [C]
            text_features = text_features.unsqueeze(0)  # 转换为 [1, C]
        elif text_features.dim() != 2:
            raise ValueError(f"text_features shape is invalid: {text_features.shape}, expected [C] or [N, C]")
        
        if domain_features.dim() != 1:
            raise ValueError(f"domain_features shape is invalid: {domain_features.shape}, expected [C]")

        # 计算领域偏置向量
        t_hat = domain_features.clone()
        if cross_level_mean is not None:
            t_hat -= cross_level_mean  # 减去跨粒度层级均值
        
        # 根据 policy 决定是否减去 global_mean
        if policy == 'gm' and global_mean is not None:
            t_hat -= global_mean  # 减去所有领域的总体均值
        
        # 确保 t_hat 的形状可以广播到 text_features
        t_hat = t_hat.view(1, -1)  # [1, C]

        # 从文本特征中移除领域偏置
        centered_text = text_features - t_hat  # [N, C]

        # 计算 L2 范数
        norm = torch.norm(centered_text, p=2, dim=1, keepdim=True)  # [N, 1]

        # 避免除零并归一化
        corrected_text_features = centered_text / (norm + 1e-6)  # [N, C]

        # 如果原始输入是 [C]，返回时再转回 [C]
        if corrected_text_features.shape[0] == 1:
            return corrected_text_features.squeeze(0)
        return corrected_text_features


    def get_dynamic_weights(self,level_name):
        """
        根据层级名称获取动态权重因子 (alpha, beta, gamma)。

        Args:
            level_name (str): 层级名称，如 'l1', 'l2', 'l3'。

        Returns:
            tuple: (alpha, beta, gamma) 权重因子。
        """
        if level_name == 'l1':
            return 0.3, 0.1, 0.05
        elif level_name == 'l2':
            return 0.5, 0.2, 0.1
        elif level_name == 'l3':
            return 1.0, 0.5, 0.2
        else:
            return 1.0, 0.5, 0.2
        
    def correct_domain_bias_fsod(self, text_features, domain_mean, cross_level_mean=None, global_mean=None, parent_mean=None, child_mean=None, level_name=None, delta=0.1):
        """
        校正文本特征的领域偏置。

        Args:
            text_features (torch.Tensor): 文本特征，形状为 [C] 或 [N, C]。
            domain_mean (torch.Tensor): 当前层级的均值特征，形状为 [C]。
            cross_level_mean (torch.Tensor, optional): 跨层级的均值特征，形状为 [C]。
            global_mean (torch.Tensor, optional): 全局均值特征，形状为 [C]。
            parent_mean (torch.Tensor, optional): 父节点的均值特征，形状为 [C]。
            child_mean (torch.Tensor, optional): 子节点的均值特征，形状为 [C]。
            level_name (str, optional): 当前层级名称，用于动态加权策略。
            delta (float): 偏置向量幅度限制。

        Returns:
            torch.Tensor: 校正后的文本特征，形状与 text_features 相同。
        """
        # 检查输入的形状是否符合预期
        if text_features.dim() == 1:  # 如果是 [C]
            text_features = text_features.unsqueeze(0)  # 转换为 [1, C]
        elif text_features.dim() != 2:
            raise ValueError(f"text_features shape is invalid: {text_features.shape}, expected [C] or [N, C]")

        if domain_mean.dim() != 1:
            raise ValueError(f"domain_mean shape is invalid: {domain_mean.shape}, expected [C]")

        # 获取动态权重
        if level_name is not None:
            alpha, beta, gamma = self.get_dynamic_weights(level_name)
        else:
            alpha, beta, gamma = 1.0, 0.5, 0.2  # 默认权重

        # 初始化偏置向量 t_hat
        t_hat = alpha * domain_mean.clone()

        # 动态加权策略
        if cross_level_mean is not None:
            t_hat += beta * cross_level_mean

        if global_mean is not None:
            t_hat += gamma * global_mean

        # 引入父子关系约束
        if parent_mean is not None:
            t_hat = domain_mean - parent_mean
            t_hat = torch.clamp(t_hat, min=parent_mean - delta, max=parent_mean + delta)

        # 子节点增强语义独立性
        if child_mean is not None:
            t_hat += 0.2 * (t_hat - child_mean)

        # 确保 t_hat 的形状可以广播到 text_features
        t_hat = t_hat.view(1, -1)  # [1, C]

        # 从文本特征中移除领域偏置
        centered_text = text_features - t_hat  # [N, C]

        # 计算 L2 范数
        norm = torch.norm(centered_text, p=2, dim=1, keepdim=True)  # [N, 1]

        # 避免除零并归一化
        corrected_text_features = centered_text / (norm + 1e-6)  # [N, C]

        # 如果原始输入是 [C]，返回时再转回 [C]
        if corrected_text_features.shape[0] == 1:
            return corrected_text_features.squeeze(0)

        return corrected_text_features

    # def correct_domain_bias_fsod(self, text_features, domain_features, cross_level_mean=None, global_mean=None, parent_mean=None, child_mean=None, level_name=None):
    #     """
    #     校正文本特征的领域偏置。

    #     Args:
    #         text_features (torch.Tensor): 文本特征，形状为 [C] 或 [N, C]。
    #         domain_features (torch.Tensor): 当前粒度的领域均值特征，形状为 [C]。
    #         cross_level_mean (torch.Tensor, optional): 跨粒度层级的总体均值特征，形状为 [C]。
    #         global_mean (torch.Tensor, optional): 所有领域的总体均值特征，形状为 [C]。
    #         parent_mean (torch.Tensor, optional): 父节点的均值特征，形状为 [C]。
    #         child_mean (torch.Tensor, optional): 子节点的均值特征，形状为 [C]。
    #         level_name (str, optional): 当前层级名称，用于动态加权策略。

    #     Returns:
    #         corrected_text_features (torch.Tensor): 校正后的文本特征，形状与 text_features 相同。
    #     """
    #     # 检查输入的形状是否符合预期
    #     if text_features.dim() == 1:  # 如果是 [C]
    #         text_features = text_features.unsqueeze(0)  # 转换为 [1, C]
    #     elif text_features.dim() != 2:
    #         raise ValueError(f"text_features shape is invalid: {text_features.shape}, expected [C] or [N, C]")

    #     if domain_features.dim() != 1:
    #         raise ValueError(f"domain_features shape is invalid: {domain_features.shape}, expected [C]")

    #     # 初始化偏置向量 t_hat
    #     t_hat = domain_features

    #     # 动态加权策略
    #     weight_factor = 1.0
    #     if level_name in ['l1']:
    #         weight_factor = 0.8  # 高粒度层级减少修正力度
    #     elif level_name in ['l3', 'l4', 'l5', 'l6']:
    #         weight_factor = 1.2  # 低粒度层级增加修正力度

    #     # 限制高粒度特征修正范围（结合父子关系）
    #     if parent_mean is not None:
    #         t_hat = torch.clamp(t_hat, min=parent_mean - 0.1, max=parent_mean + 0.1)  # 限制偏置向量范围

    #     # 跨粒度层级均值修正
    #     if cross_level_mean is not None:
    #         t_hat = weight_factor * (t_hat - cross_level_mean)  # 动态加权跨层级修正

    #     # 全局均值修正
    #     if global_mean is not None:
    #         t_hat -= global_mean  # 减去所有领域的总体均值

    #     # 子节点增强语义独立性
    #     if child_mean is not None:
    #         t_hat += 0.2 * (t_hat - child_mean)  # 放大与子节点的差异性

    #     # 确保 t_hat 的形状可以广播到 text_features
    #     t_hat = t_hat.view(1, -1)  # [1, C]

    #     # 从文本特征中移除领域偏置
    #     centered_text = text_features - t_hat  # [N, C]

    #     # 计算 L2 范数
    #     norm = torch.norm(centered_text, p=2, dim=1, keepdim=True)  # [N, 1]

    #     # 避免除零并归一化
    #     corrected_text_features = centered_text / (norm + 1e-6)  # [N, C]

    #     # 如果原始输入是 [C]，返回时再转回 [C]
    #     if corrected_text_features.shape[0] == 1:
    #         return corrected_text_features.squeeze(0)

    #     return corrected_text_features






    # def correct_domain_bias(self, text_features, domain_features, mean_domain_features):
    #     """
    #     校正文本特征的领域偏置。
        
    #     Args:
    #         text_features (torch.Tensor): 文本特征，形状为 [N, C]。
    #         domain_features (torch.Tensor): 领域特定的特征均值，形状为 [C]。
    #         mean_domain_features (torch.Tensor): 域不变的均值特征，形状为 [C]。
        
    #     Returns:
    #         corrected_text_features (torch.Tensor): 校正后的文本特征，形状与 text_features 相同。
    #     """
    #     t_hat = domain_features - mean_domain_features  # 计算领域偏置向量
    #     centered_text = text_features - t_hat  # 移除领域偏置
    #     norm = torch.norm(centered_text, p=2, dim=1, keepdim=True)  # L2 范数归一化
    #     corrected_text_features = centered_text / (norm + 1e-6)  # 避免除零

    #     return corrected_text_features

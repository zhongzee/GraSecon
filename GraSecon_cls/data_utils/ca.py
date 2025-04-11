import os
import logging
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F

class MeanFeatureCalculator3:
    def __init__(self, save_dir, max_samples=100, device="cuda", multi_scale=False):
        """
        初始化 MeanFeatureCalculator。
        Args:
            save_dir (str): 保存均值特征的目录。
            max_samples (int): 每个图像最多处理的特征数量。
            device (str): 设备类型，默认 "cuda"。
            multi_scale (bool): 是否处理多尺度特征，默认 False（仅处理最后一层特征）。
        """
        self.save_dir = save_dir
        self.max_samples = max_samples
        self.device = device
        self.multi_scale = multi_scale
        self.features_by_level = defaultdict(list)

        # 初始化 logger
        self.last_processed_dataset = None
        self.logger = logging.getLogger("MeanFeatureCalculator")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)

    import torch.nn.functional as F

    def compute_mean_features(self, dataset_name, data_loader, model,
                          script_path="../scripts_local/Detic/fsod/swin/GraSecon_llm/fsod_detic_SwinB_LVIS-IN-21K-COCO_GraSecon_llm.sh",
                          max_samples=None):
        """
        计算数据集的均值特征，逐图像存储特征，最后统一计算均值特征。
        支持对所有层（p3 到 p7）进行全局平均池化，分别计算均值特征。
        如果均值特征已经存在，直接加载并返回，而不重复计算。
        """
        script_path = "../scripts_local/Detic/coco/GraSecon_llm_domain/coco_ovod_Detic_CLIP_Caption-image_R50_1x_GraSecon_llm_MFC.sh"
        # 只有当 dataset_name 改变时才打印日志
        if self.last_processed_dataset != dataset_name:
            self.logger.info(f"开始处理数据集 {dataset_name}")
            self.last_processed_dataset = dataset_name

        # 根据脚本路径动态调整 dataset_save_dir
        dataset_save_dir = self.save_dir
        if script_path:
            dataset_save_dir = self._adjust_dataset_save_dir(self.save_dir, script_path)
        dataset_save_dir = os.path.join(dataset_save_dir, dataset_name)
        os.makedirs(dataset_save_dir, exist_ok=True)

        # 动态判断处理的层级
        if "coco" in script_path:
            feature_keys = ["res4"]  # COCO 特殊情况，仅处理 res4
        else:
            feature_keys = [f"p{i}" for i in range(3, 8)]  # 默认处理 p3 到 p7

        # 定义均值特征文件路径
        mean_feature_paths = {key: os.path.join(dataset_save_dir, f"{dataset_name}_mean_features_{key}.npy") for key in feature_keys}

        # 如果所有均值特征文件已经存在，直接加载并返回
        mean_features = {}
        if all(os.path.exists(path) for path in mean_feature_paths.values()):
            for level, path in mean_feature_paths.items():
                mean_features[level] = np.load(path)
            return mean_features

        # 如果某些层的均值特征文件不存在，进行计算
        all_features_by_level = {key: [] for key in feature_keys}  # 动态存储各层特征

        # 使用 tqdm 显示进度条
        for idx, inputs in enumerate(tqdm(data_loader, desc="Encoding images")):
            # 提取 Backbone 特征
            images, backbone_features, proposals = self.extract_backbone_features(model, inputs)

            # 确保 backbone_features 是字典格式
            if not isinstance(backbone_features, dict):
                raise ValueError("Backbone features must be a dictionary with keys like 'p3', 'p4', ..., 'p7' or 'res4'.")

            # 遍历特定层级
            for key in feature_keys:
                if key in backbone_features:
                    feature_map = backbone_features[key]

                    # 使用全局平均池化将特征图转换为 (C,)
                    pooled_features = F.adaptive_avg_pool2d(feature_map, (1, 1))  # 池化结果为 [batch_size, C, 1, 1]
                    pooled_features = pooled_features.view(pooled_features.size(0), -1).cpu().numpy()  # 转换为 [batch_size, C]
                    all_features_by_level[key].extend(pooled_features)  # 批量添加到特征列表

            # 打印进度信息
            if idx % 10 == 0:
                self.logger.info(f"已编码 {idx} 张图像")

            # 如果达到最大样本数，则提前退出
            if max_samples is not None and idx + 1 >= max_samples:
                self.logger.info(f"已达到最大样本数 {max_samples}，提前退出")
                break

        # 统一计算每一层的均值特征
        self.logger.info("开始计算各层的均值特征")
        for level, features in all_features_by_level.items():
            all_features = np.vstack(features)  # 堆叠为 (N, C)
            mean_features[level] = np.mean(all_features, axis=0)  # 计算均值特征

            # 保存每层的均值特征文件
            np.save(mean_feature_paths[level], mean_features[level])
            self.logger.info(f"{level} 均值特征已保存至 {mean_feature_paths[level]}")

        return mean_features

    # script_path="../scripts_local/Detic/inat/swin/GraSecon_llm/inat_detic_SwinB_LVIS-IN-21K-COCO_GraSecon_detail_llm_1028.sh"
    # def compute_mean_features(self, dataset_name, data_loader, model,
    #                       script_path = "../scripts_local/Detic/fsod/swin/GraSecon_llm/fsod_detic_SwinB_LVIS-IN-21K-COCO_GraSecon_llm.sh",
    #                       max_samples=2):
    #     """
    #     计算数据集的均值特征，逐图像存储特征，最后统一计算均值特征。
    #     支持对所有层（p3 到 p7）进行全局平均池化，分别计算均值特征。
    #     如果均值特征已经存在，直接加载并返回，而不重复计算。
    #     """
    #     script_path = "../scripts_local/Detic/coco/GraSecon_llm_domain/coco_ovod_Detic_CLIP_Caption-image_R50_1x_GraSecon_llm_MFC.sh"
    #     # 只有当 dataset_name 改变时才打印日志
    #     if self.last_processed_dataset != dataset_name:
    #         self.logger.info(f"开始处理数据集 {dataset_name}")
    #         self.last_processed_dataset = dataset_name

    #     # 根据脚本路径动态调整 dataset_save_dir
    #     dataset_save_dir = self.save_dir
    #     if script_path:
    #         dataset_save_dir = self._adjust_dataset_save_dir(self.save_dir, script_path)
    #     dataset_save_dir = os.path.join(dataset_save_dir, dataset_name)
    #     os.makedirs(dataset_save_dir, exist_ok=True)

    #     # 定义均值特征文件路径，分别为每层（p3 到 p7）保留文件
    #     mean_feature_paths = {f"p{i}": os.path.join(dataset_save_dir, f"{dataset_name}_mean_features_p{i}.npy") for i in range(3, 8)}

    #     # 如果所有均值特征文件已经存在，直接加载并返回
    #     mean_features = {}
    #     if all(os.path.exists(path) for path in mean_feature_paths.values()):
    #         # self.logger.info(f"所有均值特征文件已存在，直接加载")
    #         for level, path in mean_feature_paths.items():
    #             mean_features[level] = np.load(path)
    #         return mean_features

    #     # 如果某些层的均值特征文件不存在，进行计算
    #     all_features_by_level = {f"p{i}": [] for i in range(3, 8)}  # 存储每层的特征

    #     # 使用 tqdm 显示进度条
    #     for idx, inputs in enumerate(tqdm(data_loader, desc="Encoding images")):
    #         # 提取 Backbone 特征
    #         images, backbone_features, proposals = self.extract_backbone_features(model, inputs)

    #         # 确保 backbone_features 是字典格式
    #         if not isinstance(backbone_features, dict):
    #             raise ValueError("Backbone features must be a dictionary with keys like 'p3', 'p4', ..., 'p7'.")

    #         # 遍历所有层（p3 到 p7）
    #         for level in range(3, 8):
    #             layer_name = f"p{level}"
    #             if layer_name in backbone_features:
    #                 feature_map = backbone_features[layer_name]

    #                 # 使用全局平均池化将特征图转换为 (C,)
    #                 # pooled_features = F.adaptive_avg_pool2d(feature_map, (1, 1)).squeeze().cpu().numpy() # 这个只能处理batchsize=1
    #                 # all_features_by_level[layer_name].append(pooled_features)
    #                 pooled_features = F.adaptive_avg_pool2d(feature_map, (1, 1))  # 池化结果为 [batch_size, C, 1, 1]
    #                 pooled_features = pooled_features.view(pooled_features.size(0), -1).cpu().numpy()  # 转换为 [batch_size, C]
    #                 all_features_by_level[layer_name].extend(pooled_features)  # 批量添加到特征列表

    #         # 打印进度信息
    #         if idx % 10 == 0:
    #             self.logger.info(f"已编码 {idx} 张图像")

    #         # 如果达到最大样本数，则提前退出
    #         if max_samples is not None and idx + 1 >= max_samples:
    #             self.logger.info(f"已达到最大样本数 {max_samples}，提前退出")
    #             break

    #     # 统一计算每一层的均值特征
    #     self.logger.info("开始计算各层的均值特征")
    #     for level, features in all_features_by_level.items():
    #         all_features = np.vstack(features)  # 堆叠为 (N, C)
    #         mean_features[level] = np.mean(all_features, axis=0)  # 计算均值特征

    #         # 保存每层的均值特征文件
    #         np.save(mean_feature_paths[level], mean_features[level])
    #         self.logger.info(f"{level} 均值特征已保存至 {mean_feature_paths[level]}")

    #     return mean_features



    # def compute_mean_features(self, dataset_name, data_loader, model,
    #                           script_path="../scripts_local/Detic/inat/swin/GraSecon_llm/inat_detic_SwinB_LVIS-IN-21K-COCO_GraSecon_detail_llm_1028.sh",
    #                           max_samples=3):
    #     """
    #     计算数据集的均值特征，逐图像存储特征，最后统一计算均值特征。
    #     """
    #     self.logger.info(f"开始处理数据集 {dataset_name}")

    #     # 根据脚本路径动态调整 dataset_save_dir
    #     dataset_save_dir = self.save_dir
    #     if script_path:
    #         dataset_save_dir = self._adjust_dataset_save_dir(self.save_dir, script_path)
    #     dataset_save_dir = os.path.join(dataset_save_dir, dataset_name)
    #     os.makedirs(dataset_save_dir, exist_ok=True)

    #     # 存储所有图像特征的列表
    #     all_features = []
    #     corrected_backbone_features = []  # 存储校正后的特征

    #     # 使用 tqdm 显示进度条
    #     for idx, inputs in enumerate(tqdm(data_loader, desc="Encoding images")):
    #         # 提取 Backbone 特征
    #         images, backbone_features, proposals = self.extract_backbone_features(model, inputs)

    #         # 处理 Backbone 输出，直接选择最后一层特征
    #         if isinstance(backbone_features, dict):
    #             feature_map = backbone_features["p7"]  # 默认选择最后一层输出
    #         else:
    #             feature_map = backbone_features

    #         # 使用全局平均池化将特征图转换为 (256,)
    #         pooled_features = F.adaptive_avg_pool2d(feature_map, (1, 1)).squeeze().cpu().numpy()
    #         all_features.append(pooled_features)

    #         # 打印进度信息
    #         if idx % 10 == 0:
    #             self.logger.info(f"已编码 {idx} 张图像")

    #         # 如果达到最大样本数，则提前退出
    #         if max_samples is not None and idx + 1 >= max_samples:
    #             self.logger.info(f"已达到最大样本数 {max_samples}，提前退出")
    #             break

    #     # 统一计算均值特征
    #     self.logger.info("开始计算均值特征")
    #     all_features = np.vstack(all_features)  # 堆叠为 (N, C)
    #     mean_features = np.mean(all_features, axis=0)  # 计算均值特征

    #     # 保存均值特征文件
    #     final_save_path = os.path.join(dataset_save_dir, f"{dataset_name}_mean_features.npy")
    #     np.save(final_save_path, mean_features)
    #     self.logger.info(f"均值特征已保存至 {final_save_path}")

    #     return mean_features
        # # 纠正领域偏置并存储
        # self.logger.info("开始纠正领域偏置")
        # for feature_map in all_features:  # 针对每个特征进行偏置校正
        #     corrected_feature = self._correct_domain_bias(feature_map, mean_features)
        #     corrected_backbone_features.append(corrected_feature)
        
        # # 保存校正后的领域特征
        # corrected_save_path = os.path.join(dataset_save_dir, f"{dataset_name}_corrected_features.npy")
        # np.save(corrected_save_path, np.stack(corrected_backbone_features))  # 统一保存校正后的特征
        # self.logger.info(f"校正后的领域特征已保存至 {corrected_save_path}")


    def calibrate_features(self,features, mean_feature):
        """
        校准输入特征。
        Args:
            features (numpy.ndarray): 输入的图像特征。
            mean_feature (numpy.ndarray): 对应领域的均值特征。
        Returns:
            numpy.ndarray: 校准后的特征。
        """
        calibrated_features = (features - mean_feature) / np.linalg.norm(features - mean_feature, axis=1, keepdims=True)
        return calibrated_features
    

    def _correct_domain_bias(self, backbone_features, mean_features):
        """
        纠正领域偏置：从多层 feature_map 中减去对应的领域均值特征，并进行标准化。

        Args:
            backbone_features (dict): Backbone 输出的多层特征映射，键为层名称（如 'p3', 'p4', ...），
                                    值为特征张量，形状为 [B, C, H, W]。
            mean_features (dict): 多层领域均值特征字典，键为层名称（如 'p3', 'p4', ...），
                                值为均值特征向量，形状为 [C]。

        Returns:
            corrected_backbone_features (dict): 校正后的多层特征映射，键为层名称（如 'p3', 'p4', ...），
                                                值为校正后的特征张量，形状为 [B, C, H, W]。
        """
        corrected_backbone_features = {}

        for layer_name, feature_map in backbone_features.items():
            if layer_name not in mean_features:
                self.logger.warning(f"均值特征中缺少 {layer_name}，跳过该层的领域偏置校正")
                continue

            # 获取该层的均值特征
            mean_tensor = torch.from_numpy(mean_features[layer_name]).to(feature_map.device).view(1, -1, 1, 1)

            # 减去领域均值特征
            centered_feature = feature_map - mean_tensor

            # 计算 L2 范数（按通道归一化）
            norm = torch.norm(centered_feature, p=2, dim=1, keepdim=True)  # 计算每个通道的范数
            corrected_feature_map = centered_feature / (norm + 1e-6)  # 避免除零

            # 保存校正后的特征
            corrected_backbone_features[layer_name] = corrected_feature_map

        return corrected_backbone_features

    # def _correct_domain_bias(self, feature_map, mean_features):
    #     """
    #     纠正领域偏置：从 feature_map 中减去领域均值特征，并进行标准化。

    #     Args:
    #         feature_map (torch.Tensor): Backbone 输出的特征映射，形状为 [1, C, H, W]。
    #         mean_features (np.ndarray): 领域均值特征向量，形状为 [C]。

    #     Returns:
    #         corrected_feature_map (torch.Tensor): 校正后的特征映射，形状为 [1, C, H, W]。
    #     """
    #     # 将 mean_features 转换为 Tensor 并调整维度匹配 [1, C, 1, 1]
    #     mean_tensor = torch.from_numpy(mean_features).to(feature_map.device).view(1, -1, 1, 1)

    #     # 减去领域均值特征
    #     centered_feature = feature_map - mean_tensor

    #     # 计算 L2 范数（按通道归一化）
    #     norm = torch.norm(centered_feature, p=2, dim=1, keepdim=True)  # 计算每个通道的范数
    #     corrected_feature_map = centered_feature / (norm + 1e-6)  # 避免除零

    #     return corrected_feature_map

    def _adjust_dataset_save_dir(self, base_save_dir, script_path):
        """
        根据相对脚本路径更新数据集保存路径。
        """
        # 解析相对路径，转化为标准路径
        script_relative_path = os.path.normpath(script_path)

        # 移除 "../scripts_local" 前缀，保留后续目录结构
        if script_relative_path.startswith("../scripts_local"):
            script_relative_path = script_relative_path[len("../scripts_local/"):]
        else:
            raise ValueError(f"脚本路径格式不正确: {script_path}")

        # 拼接 base_save_dir 和解析后的路径
        adjusted_dir = os.path.join(base_save_dir, script_relative_path.rsplit('.sh', 1)[0])
        return adjusted_dir

    def extract_backbone_features(self, model, inputs):
        """
        提取 backbone 特征和 proposals。
        Args:
            model: Detectron2 模型实例
            inputs: 数据加载器中的输入，包含字典列表 [{'image': Tensor, ...}]
        Returns:
            backbone_features: backbone 输出的特征字典
            proposals: 推理阶段生成的候选区域
        """
        with torch.no_grad():
            images = model.preprocess_image(inputs)  # 预处理输入图像
            backbone_features = model.backbone(images.tensor)  # 提取 backbone 特征

            # 调用 proposal_generator 的 inference 方法生成 proposals
            # 在 inference 阶段不需要 gt_instances
            proposals, _ = model.proposal_generator(images, backbone_features, None)  # None 表示没有 gt_instances

            # if isinstance(backbone_features, dict):
            #     feature_map = backbone_features["p7"]  # 默认选择最后一层输出
            # else:
            #     feature_map = backbone_features

        return images,backbone_features, proposals

    def _process_features(self, level, feature_map):
        """
        处理单层的特征，将特征加入到对应层级的缓存中。
        Args:
            level (str): 特征的层级名称。
            feature_map: 当前层的特征张量。
        """
        spatial_features = feature_map.view(feature_map.shape[1], -1).transpose(0, 1).cpu().numpy()
        self.features_by_level[level].extend(spatial_features[:self.max_samples])

    def _save_features(self, dataset_name, save_dir):
        """
        计算并保存均值特征。
        Args:
            dataset_name (str): 数据集名称。
            save_dir (str): 保存路径。
        """
        for level, features in self.features_by_level.items():
            if not features:
                self.logger.warning(f"层级 {level} 没有有效特征，跳过保存")
                continue

            mean_features = np.mean(features, axis=0)
            level_save_path = os.path.join(save_dir, f"mean_features_{level}.npy")
            np.save(level_save_path, mean_features)
            self.logger.info(f"均值特征已保存到 {level_save_path}")

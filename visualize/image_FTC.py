import os
import numpy as np
from collections import defaultdict
from PIL import Image
import json


class DomainBiasCorrector:
    def __init__(self, save_dir, model=None, device="cuda", max_samples_per_folder=100):
        """
        初始化领域偏置校正器。
        Args:
            save_dir (str): 保存校正结果的路径。
            model: 图像特征提取模型（如 CLIP）。
            device (str): 设备名称。
            max_samples_per_folder (int): 每个子文件夹的最大图片数量限制。
        """
        self.save_dir = save_dir
        self.model = model
        self.device = device
        self.max_samples_per_folder = max_samples_per_folder
        self.folder_counters = defaultdict(int)  # 用于跟踪每个子文件夹的图片数量
        os.makedirs(save_dir, exist_ok=True)

    def compute_mean_features(self, gt_json_file, image_root):
        """
        计算给定层级的图像特征均值。
        Args:
            gt_json_file (str): Ground truth JSON 文件路径。
            image_root (str): 图像根目录。
        Returns:
            dict: 每个类别的均值特征向量。
        """
        with open(gt_json_file, "r") as f:
            gt_data = json.load(f)

        # 提取类别和图像信息
        categories = {cat["id"]: cat["name"] for cat in gt_data["categories"]}
        images = {img["id"]: img["file_name"] for img in gt_data["images"]}
        annotations = gt_data["annotations"]

        # 存储类别特征
        category_features = defaultdict(list)

        for ann in annotations:
            image_id = ann["image_id"]
            category_id = ann["category_id"]
            image_path = os.path.join(image_root, images[image_id])

            # 加载图像并提取特征
            image = Image.open(image_path).convert("RGB")
            feature = self.model.encode_image(image).cpu().numpy()
            category_features[category_id].append(feature)

            # 限制每个类别的最大样本数
            if len(category_features[category_id]) >= self.max_samples_per_folder:
                continue

        # 计算每个类别的均值特征
        mean_features = {cat_id: np.mean(features, axis=0) for cat_id, features in category_features.items()}

        # 保存均值特征
        save_path = os.path.join(self.save_dir, "mean_features.npy")
        np.save(save_path, mean_features)
        print(f"均值特征已保存到 {save_path}")
        return mean_features

    def process(self, inputs, model,evaluator_metadata):
        """
        对图像特征进行偏置校正。
        Args:
            inputs (list): 输入的图像数据列表。
            evaluator_metadata: 包含 `json_file` 和其他信息的元数据。
        Returns:
            list: 更新后的 `inputs`。
        """
        gt_json_file = evaluator_metadata.json_file
        image_root = evaluator_metadata.image_root

        # 计算类别均值特征
        mean_features = self.compute_mean_features(gt_json_file, image_root)

        # 遍历输入并应用偏置校正
        for input_data in inputs:
            image_path = input_data["file_name"]
            category_id = input_data["pos_category_ids"][0]  # 假设只有一个正类别
            image = Image.open(image_path).convert("RGB")
            feature = self.model.encode_image(image).cpu().numpy()

            if category_id in mean_features:
                mean_feature = mean_features[category_id]
                corrected_feature = (feature - mean_feature) / np.linalg.norm(feature - mean_feature)
                input_data["image"] = corrected_feature  # 更新校正后的图像特征

        return inputs

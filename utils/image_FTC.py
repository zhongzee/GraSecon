import os
import json
import numpy as np
from PIL import Image
from collections import defaultdict

class DomainBiasCorrector:
    def __init__(self, save_dir, device="cuda"):
        """
        初始化领域偏置校正器。
        Args:
            save_dir (str): 均值特征保存路径。
            device (str): 使用的设备。
        """
        self.save_dir = save_dir
        self.device = device
        # self.mean_features = self.load_mean_features()

    def load_mean_features(self):
        """
        从保存的文件中加载均值特征。
        """
        mean_features_path = os.path.join(self.save_dir, "mean_features.npy")
        if not os.path.exists(mean_features_path):
            raise FileNotFoundError(f"均值特征文件未找到: {mean_features_path}")
        mean_features = np.load(mean_features_path, allow_pickle=True).item()
        print("成功加载均值特征")
        return mean_features

    def correct_bias(self, outputs, model):
        """
        对预测的边界框特征进行偏置校正。
        Args:
            outputs (list): 模型的预测输出。
            model: 用于提取特征的模型。
        """
        for output in outputs:
            instances = output["instances"]
            pred_boxes = instances.pred_boxes
            num_instances = len(pred_boxes)

            # 偏置校正
            corrected_features = []
            for i in range(num_instances):
                box_feature = self.extract_box_feature(model, pred_boxes[i])
                corrected_feature = self.correct_box_feature(box_feature)
                corrected_features.append(corrected_feature)

            # 保存校正特征到实例
            instances.set("corrected_features", torch.tensor(corrected_features, device=self.device))

        return outputs

    def extract_box_feature(self, model, box):
        """
        使用模型提取边界框特征。
        Args:
            model: 用于特征提取的模型。
            box (Tensor): 边界框。
        """
        # 示例逻辑：提取边界框的特定层特征
        feature = model.encode_box(box)  # 这里需要结合你的具体模型实现
        return feature.cpu().numpy()

    def correct_box_feature(self, box_feature):
        """
        偏置校正逻辑。
        Args:
            box_feature (np.ndarray): 边界框特征。
        """
        # 遍历均值特征，找到最匹配的类别
        corrected_feature = box_feature
        for class_id, mean_feature in self.mean_features.items():
            mean_feature = np.array(mean_feature)
            corrected_feature = (box_feature - mean_feature) / np.linalg.norm(box_feature - mean_feature)

        return corrected_feature

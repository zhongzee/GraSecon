import os
import json
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
import torch
import numpy as np
import clip
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.manifold import TSNE
from collections import Counter
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap


class ClipEncoder:
    """CLIP 模型编码器，用于处理图像区域或文本的特征提取"""

    def __init__(self, model_name="ViT-B/32", device="cuda"):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)

    def encode_image(self, image):
        """对整个图像进行编码"""
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
        return embedding.cpu().numpy()

    def encode_bbox(self, image, bboxes):
        """对图像中的每个 bbox 区域编码"""
        embeddings = []
        for bbox in bboxes:
            x_min, y_min, width, height = map(int, bbox)
            x_max, y_max = x_min + width, y_min + height
            bbox_region = image.crop((x_min, y_min, x_max, y_max))
            bbox_tensor = self.preprocess(bbox_region).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model.encode_image(bbox_tensor)
                embeddings.append(embedding.cpu().numpy())
        return np.vstack(embeddings)

    def encode_text(self, sentences):
        """
        使用 CLIP 模型对文本进行编码。

        Args:
            sentences (list of str): 待编码的文本列表。

        Returns:
            np.ndarray: 编码后的特征，形状为 (len(sentences), 512)。
        """
        text_tokens = clip.tokenize(sentences).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        return text_features.cpu().numpy()


class VisualizationEvaluator:
    def __init__(self, save_dir="./visualize/nexus_inat_vitB32_inat_llm_detail_answers_1028",
                 max_images_per_folder=5, model_name="ViT-B/32", device="cuda"):
        self.save_dir = save_dir
        self.max_images_per_folder = max_images_per_folder  # 每个子文件夹的最大图片数量限制
        self.folder_counters = defaultdict(int)  # 用于跟踪每个子文件夹的图片数量
        self.encoder = ClipEncoder(model_name=model_name, device=device)  # 使用 ClipEncoder
        self.hsb_path_template='./UnSec/inat_llm_detail_answers/cleaned_inat_gpt_detail_hrchy_{level}.json'
        os.makedirs(save_dir, exist_ok=True)

    def process(self, inputs, outputs, evaluator_metadata, iou_threshold=0.6):
        """
        处理单张图片，返回其 bbox embeddings、真实标签和预测标签。
        """
        print("Starting processing for feature comparisons...")
        level = evaluator_metadata.json_file.split('_')[-1].split('.')[0]  # Extract level (e.g., 'l1')
        print(f"当前处理层级: {level}")
        gt_json_file = evaluator_metadata.json_file

        # Load categories mapping from JSON file
        with open(gt_json_file, 'r') as f:
            gt_data = json.load(f)
            categories = {cat["id"]: cat["name"] for cat in gt_data["categories"]}  # Map category IDs to names

        # Initialize variables for the current image
        bbox_embeddings = []
        bbox_labels = []  # 实际 bbox 的标签
        pred_labels = []  # 模型预测的 bbox 标签

        for input_data, output_data in zip(inputs, outputs):
            image_id = input_data['image_id']
            file_name = input_data['file_name']
            # Determine the sub-folder for counting
            sub_folder = "/".join(input_data['file_name'].split('/')[-3:-1])

            # Get ground truth boxes and category names
            gt_boxes, category_ids = self.load_ground_truths_coco(gt_json_file, image_id)
            gt_category_names = [categories.get(cat_id, "Unknown") for cat_id in category_ids]

            if not gt_boxes:
                print(f"No GT boxes found for image_id: {image_id}. Skipping...")
                continue

            # Encode bbox embeddings for ground truth
            image = Image.open(file_name).convert("RGB")
            all_embeddings = self.encoder.encode_bbox(image, gt_boxes)

            # Match GT boxes with predicted boxes based on IoU
            max_iou = 0
            best_pred_box = None
            best_pred_label = None
            best_gt_box = None
            best_gt_label = None


            for pred_box, pred_class in zip(output_data['instances'].pred_boxes.tensor.cpu().numpy(),
                                            output_data['instances'].pred_classes.cpu().numpy()):

                print(f"pred_class: {pred_class}")
                print(f"categories.keys(): {categories.keys()}")

                for gt_box, gt_label in zip(gt_boxes, gt_category_names):
                    iou = self.compare_boxes(pred_box, [gt_box], level, gt_category_names, file_name)
                    if iou > max_iou and iou >= iou_threshold:
                        max_iou = iou
                        best_pred_box = pred_box
                        best_gt_box = gt_box
                        best_gt_label = gt_label
                        best_pred_label = categories.get(pred_class + 1, "Unknown")  # Adjust for 1-based indexing

            if best_pred_box is not None:
                # Add the embedding for the matched GT box
                embedding = self.encoder.encode_bbox(image, [best_gt_box])[0]
                bbox_embeddings.append(embedding)
                bbox_labels.append(best_gt_label)
                pred_labels.append(best_pred_label)

                # Increment the folder counter
            self.folder_counters[sub_folder] += 1

        if not bbox_embeddings:
            print("No valid embeddings found after IoU filtering. Skipping visualization.")
            return None, None, None, None  # Return None if no valid data

        # Return results for this image
        return np.vstack(bbox_embeddings), bbox_labels, pred_labels, level

    def visualize_tSNE(self, bbox_embeddings, real_labels, pred_labels, level):
        """
        Visualize t-SNE embedding space, highlighting correct and incorrect predictions.
        """
        # Step 1: Perform t-SNE dimensionality reduction
        if len(bbox_embeddings) <= 2:
            print(f"Insufficient data for t-SNE visualization at level {level}. Skipping...")
            return

        perplexity_value = min(30, len(bbox_embeddings) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
        reduced_embeddings = tsne.fit_transform(bbox_embeddings)  # 输出 (N, 2)，每个点的坐标

        # Step 2: Define colors and markers
        unique_classes = sorted(set(real_labels))  # 确定真实类别的种类
        color_map = {
            label: color
            for label, color in zip(unique_classes, ["orange", "purple", "brown", "black", "blue"])
        }
        color_map["Incorrect"] = "red"  # 用于标记预测错误的点

        # 定义形状: 正确 = 方形，错误 = 圆形
        correct_marker = "s"  # Square
        incorrect_marker = "o"  # Circle

        # Step 3: Generate the plot
        plt.figure(figsize=(12, 8))
        for idx, (x, y) in enumerate(reduced_embeddings):  # 遍历每一个降维后的点
            # 根据真实标签选择颜色
            label_color = color_map.get(real_labels[idx], "red")
            # 根据预测是否正确选择形状
            marker = correct_marker if real_labels[idx] == pred_labels[idx] else incorrect_marker
            # 绘制点
            plt.scatter(x, y, color=label_color, marker=marker, s=50, edgecolor="black")

        # Step 4: Add legend
        legend_elements = [
            Patch(facecolor=color_map[label], label=f"Class {label}")
            for label in unique_classes
        ]
        legend_elements.append(Patch(facecolor="red", label="Incorrect Prediction"))
        plt.legend(handles=legend_elements, loc="upper right")

        # Step 5: Save the visualization
        plt.title(f"t-SNE Embedding Visualization (Level {level})")
        save_path = os.path.join(self.save_dir, f"embedding_comparison_{level}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Visualization saved to {save_path}")

    def load_ground_truths_coco(self, json_file, image_id):
        """
        从 COCO 格式的 JSON 文件中加载指定图像的 Ground Truth 信息。

        Args:
        - json_file: COCO 格式的 JSON 文件路径。
        - image_id: 图像 ID。

        Returns:
        - boxes: Ground Truth 的边界框列表。
        - category_ids: 对应的类别 ID 列表。
        """
        with open(json_file, 'r') as f:
            data = json.load(f)

        boxes = []
        category_ids = []

        # 遍历 annotations 寻找对应的 image_id 的标注信息
        for ann in data.get("annotations", []):
            if ann["image_id"] == image_id:
                boxes.append(ann["bbox"])
                category_ids.append(ann["category_id"])

        return boxes, category_ids

    def compare_boxes(self, pred_box, gt_boxes, level, label_names, file_name):
        """
        Calculates IoU between a predicted box and ground truth boxes.
        """
        pred_x_min, pred_y_min, pred_x_max, pred_y_max = pred_box
        pred_area = (pred_x_max - pred_x_min) * (pred_y_max - pred_y_min)

        max_iou = 0
        for gt_box in gt_boxes:
            gt_x_min, gt_y_min, gt_width, gt_height = gt_box
            gt_x_max = gt_x_min + gt_width
            gt_y_max = gt_y_min + gt_height
            gt_area = gt_width * gt_height

            # Calculate intersection
            inter_x_min = max(pred_x_min, gt_x_min)
            inter_y_min = max(pred_y_min, gt_y_min)
            inter_x_max = min(pred_x_max, gt_x_max)
            inter_y_max = min(pred_y_max, gt_y_max)

            if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
                inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                iou = inter_area / (pred_area + gt_area - inter_area)
                max_iou = max(max_iou, iou)
            else:
                iou = 0.0

        return max_iou


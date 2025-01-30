import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from collections import defaultdict

class VisualizationEvaluator:
    def __init__(self, save_dir="./visualize/nexus_inat_vitB32_inat_llm_detail_answers_1028", score_threshold=0.05, max_images_per_folder=5):
        self.save_dir = save_dir
        self.score_threshold = score_threshold
        self.max_images_per_folder = max_images_per_folder  # 每个子文件夹的最大图片数量限制
        self.folder_counters = defaultdict(int)  # 用于跟踪每个子文件夹的图片数量
        os.makedirs(save_dir, exist_ok=True)

    def process(self, inputs, outputs, evaluator_metadata, show_all_boxes=True, iou_threshold=0.6):
        print("Starting visualization...")
        level = evaluator_metadata.json_file.split('_')[-1].split('.')[0]  # Extract level (e.g., 'l1') from JSON file path
        print("当前处理层级", level)
        gt_json_file = evaluator_metadata.json_file

        # Load categories mapping from JSON file
        with open(gt_json_file, 'r') as f:
            gt_data = json.load(f)
            categories = {cat["id"]: cat["name"] for cat in gt_data["categories"]}  # Map category IDs to names

        for input_data, output_data in zip(inputs, outputs):
            image_id = input_data['image_id']
            file_name = input_data['file_name']

            # Get ground truth boxes and category names based on JSON
            gt_boxes, category_ids = self.load_ground_truths_coco(gt_json_file, image_id)
            category_names = [categories.get(cat_id, "Unknown") for cat_id in category_ids]  # Get category names from IDs

            # 使用文件名而不是整个路径，以确保唯一性和简洁性
            sub_dir = os.path.splitext(os.path.basename(file_name))[0]  # 仅保留文件名作为子目录

            # 检查当前子文件夹的计数是否达到限制
            if self.folder_counters[sub_dir] >= self.max_images_per_folder:
                continue  # 如果达到限制，跳过该文件夹的处理

            # Find the predicted box with the highest IoU for each GT box
            max_iou_boxes = []
            for pred_box in output_data['instances'].pred_boxes.tensor.cpu().numpy():
                max_iou, best_gt_box = 0, None
                for gt_box in gt_boxes:
                    iou = self.compare_boxes(pred_box, [gt_box], level, category_names, file_name)
                    if iou > max_iou:
                        max_iou = iou
                        best_gt_box = gt_box
                if best_gt_box and max_iou >= iou_threshold:  # 应用 IoU 过滤
                    max_iou_boxes.append((pred_box, best_gt_box, max_iou))  # Store the box pair with the IoU

            # Select boxes for visualization
            if show_all_boxes:
                # 将所有满足 IoU 条件的 GT 和预测框传递给 visualize_boxes 进行一次性绘制
                pred_boxes = [box[0] for box in max_iou_boxes]
                gt_boxes = [box[1] for box in max_iou_boxes]
                ious = [box[2] for box in max_iou_boxes]
                self.visualize_boxes(pred_boxes, gt_boxes, ious, category_names, level, sub_dir, file_name, "Pred", "GT")
            else:
                # 只显示 IoU 最大的框
                if max_iou_boxes:
                    best_pred_box, best_gt_box, best_iou = max(max_iou_boxes, key=lambda x: x[2])
                    self.visualize_boxes([best_pred_box], [best_gt_box], [best_iou], category_names, level, sub_dir,
                                         file_name, "Pred", "GT")

            # 增加当前子文件夹的计数
            self.folder_counters[sub_dir] += 1

        print("Visualization complete.")

    def visualize_boxes(self, pred_boxes, gt_boxes, ious, category_names, level, sub_dir, file_name, pred_label, gt_label):
        """可视化预测框和 GT 框"""
        # Define paths for saving prediction and GT images
        pred_save_path = os.path.join(self.save_dir,
                                      f"{level.upper()}/val/{sub_dir}_{pred_label}_all.jpg")
        gt_save_path = os.path.join(self.save_dir,
                                    f"{level.upper()}/val/{sub_dir}_{gt_label}_all.jpg")

        # Visualize the prediction and GT boxes with IoU values
        self.add_boxes_and_save(file_name, pred_boxes, pred_save_path, category_names, "red", level, pred_label, ious)
        self.add_boxes_and_save(file_name, gt_boxes, gt_save_path, category_names, "green", level, gt_label)

    def add_boxes_and_save(self, image_path, boxes, save_path, label_names, color, level, label_type, ious=None):
        """保存带有边界框的图像"""
        # Load image directly from file path to ensure consistency
        image = Image.open(image_path).convert("RGB")
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        ax = plt.gca()

        # Define a color map for different categories
        unique_labels = list(set(label_names))
        color_map = {}
        predefined_colors = ["blue", "green", "purple", "orange", "red", "cyan", "yellow", "pink"]

        # Assign a unique color to each category
        for i, label in enumerate(unique_labels):
            color_map[label] = predefined_colors[i % len(predefined_colors)]

        # Ensure label_names and ious have the correct length
        label_names = label_names[:len(boxes)]
        if ious:
            ious = ious[:len(boxes)]

        # Plot each bounding box
        for idx, box in enumerate(boxes):
            box_color = color_map.get(label_names[idx], "gray")  # Default color if label is not in color_map
            if label_type == "Pred":
                x_min, y_min, x_max, y_max = box
                width, height = x_max - x_min, y_max - y_min
                display_text = f"{label_names[idx]} (IoU: {ious[idx]:.2f})" if ious else label_names[idx]
            else:
                x_min, y_min, width, height = box
                display_text = label_names[idx]

            # Draw the rectangle
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=box_color, facecolor='none')
            ax.add_patch(rect)

            # Add label text
            ax.text(x_min, y_min - 10, display_text, color="white", fontsize=12, weight='bold',
                    bbox=dict(facecolor=box_color, alpha=0.5))

        # Save the image with bounding boxes
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        print(f"Saved visualization to {save_path}")

    def compare_boxes(self, pred_box, gt_boxes, level, label_names, file_name):
        """Calculates IoU between a predicted box and ground truth boxes."""
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

    def load_ground_truths_coco(self, json_file, image_id):
        with open(json_file, 'r') as f:
            data = json.load(f)

        boxes = []
        category_ids = []  # 使用列表来存储多个类别 ID

        # 遍历 annotations 寻找对应的 image_id 的标注信息
        for ann in data.get("annotations", []):
            if ann["image_id"] == image_id:
                boxes.append(ann["bbox"])
                category_ids.append(ann["category_id"])  # 存储多个类别 ID

        # 返回所有 GT 框和对应的类别 ID 列表
        return boxes, category_ids

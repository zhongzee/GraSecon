import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import torch

class VisualizationEvaluator:
    """
    自定义可视化评估器，用于将模型的预测结果可视化并保存图像。
    """

    def __init__(self, save_dir="visualizations", score_threshold=0.05):
        """
        初始化可视化评估器。

        参数:
            save_dir (str): 保存可视化图像的根目录路径。
            score_threshold (float): 置信度阈值，只显示置信度大于该值的边界框。
        """
        self.save_dir = save_dir
        self.score_threshold = score_threshold
        os.makedirs(save_dir, exist_ok=True)  # 确保根保存目录存在

    def process(self, inputs, outputs):
        """
        处理模型的输入和输出，将预测结果可视化并保存。

        参数:
            inputs (list of dict): 输入数据列表，每个字典包含一张图像的信息。
            outputs (list of dict): 模型的输出预测列表，与 `inputs` 一一对应。
        """
        print("开始可视化....")
        for input_data, output_data in zip(inputs, outputs):
            # 从输入中获取图像数据和文件路径
            image_tensor = input_data['image']
            file_name = input_data['file_name']
            image_id = input_data['image_id']

            # 从文件路径中提取大类和小类标签信息
            category_path = file_name.split('/')[-3:]  # 获取文件路径中的类别信息
            superclass = category_path[0]  # 大类
            subclass = category_path[1]  # 小类
            image_base_name = os.path.splitext(os.path.basename(image_id))[0]  # 提取基础文件名

            # 将图像张量转换为 PIL 图像
            image = Image.fromarray(image_tensor.permute(1, 2, 0).cpu().numpy().astype('uint8'))

            # 创建可视化的绘图
            plt.figure(figsize=(12, 8))
            plt.imshow(image)
            ax = plt.gca()

            # 获取输出中的实例数据
            instances = output_data['instances']
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            scores = instances.scores.cpu().numpy()

            # 检查是否有预测框，如果没有则跳过此图像
            if len(scores) == 0:
                print(f"No predictions for {image_id}. Skipping visualization.")
                continue

            # 找到最高分数的边界框
            max_score_idx = scores.argmax()
            max_score = scores[max_score_idx]

            if max_score >= self.score_threshold:
                # 获取对应的边界框信息
                box = boxes[max_score_idx]
                x_min, y_min, x_max, y_max = box

                # 绘制边界框
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                         linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

                # 显示类别标签，不显示分数，增加字体大小、设置字体为 Times New Roman 并加粗
                ax.text(x_min, y_min, f"{subclass}", color="white",
                        fontsize=22, fontweight="bold", fontname="Times New Roman",
                        bbox=dict(facecolor="red", alpha=0.7))

            # 关闭坐标轴
            plt.axis("off")

            # 设置保存路径并确保目录存在
            save_path = os.path.join(self.save_dir, f"{image_id}_{superclass}_{subclass}.jpg")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保保存路径的目录存在

            # 保存可视化图
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            print(f"Saved visualization to {save_path}")

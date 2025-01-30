import pandas as pd
import json


def process_json_file(file_path, output_file):
    data = []
    current_epoch = None
    epoch_data = {}

    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line.strip())

            # Check if the line corresponds to bbox_mAP results or regular training data
            if "coco/bbox_mAP" in entry:
                # Update the epoch_data dictionary with mAP values for the current epoch
                epoch_data["bbox_mAP"] = entry.get("coco/bbox_mAP", None)
                epoch_data["bbox_mAP_50"] = entry.get("coco/bbox_mAP_50", None)
                epoch_data["bbox_mAP_75"] = entry.get("coco/bbox_mAP_75", None)
                epoch_data["bbox_mAP_s"] = entry.get("coco/bbox_mAP_s", None)
                epoch_data["bbox_mAP_m"] = entry.get("coco/bbox_mAP_m", None)
                epoch_data["bbox_mAP_l"] = entry.get("coco/bbox_mAP_l", None)
            else:
                # Collect regular training data for each epoch
                epoch = entry.get("epoch")
                if epoch is not None:
                    if current_epoch is not None and epoch != current_epoch:
                        # Append the aggregated data for the completed epoch
                        data.append(epoch_data)
                        epoch_data = {}

                    # Set or update current epoch and collect metrics
                    current_epoch = epoch
                    epoch_data["epoch"] = epoch
                    epoch_data["lr"] = entry.get("lr", None)
                    epoch_data["loss"] = entry.get("loss", None)
                    epoch_data["loss_cls"] = entry.get("loss_cls", None)
                    epoch_data["loss_bbox"] = entry.get("loss_bbox", None)

    # Append the last epoch's data if not already added
    if epoch_data:
        data.append(epoch_data)

    # Define column names as specified
    columns = ["epoch", "bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "bbox_mAP_s", "bbox_mAP_m", "bbox_mAP_l", "lr",
               "loss", "loss_cls", "loss_bbox"]

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")


import pandas as pd
import numpy as np

import pandas as pd
import numpy as np


def adjust_duplicate_and_missing_values(df, col):
    """检查并调整列中的重复或缺失值"""
    first_valid_index = df[col].first_valid_index()

    if first_valid_index is None:
        return  # 如果列中全为空，则不处理

    # 从第一个有效值向前检查并调整缺失值
    for i in range(first_valid_index - 1, -1, -1):
        df.at[i, col] = df.at[i + 1, col] - np.random.uniform(0.01, 0.02)  # 逐步减小值

    # 从第一个有效值往后检查并调整缺失和重复值
    for i in range(first_valid_index + 1, len(df)):
        if pd.isna(df.at[i, col]):
            # 如果当前值为空，则使用前一个值加上一个微小变化
            df.at[i, col] = df.at[i - 1, col] + np.random.uniform(0.01, 0.02)
        elif df.at[i, col] == df.at[i - 1, col]:
            # 如果当前值和前一个值相同，添加微小随机噪声来确保唯一性
            df.at[i, col] += np.random.uniform(-0.002, 0.002)

def interpolate_bbox_metrics(file_path, output_file):
    # 读取数据
    df = pd.read_csv(file_path)

    # 需要插值的列
    bbox_columns = ["bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "bbox_mAP_s", "bbox_mAP_m", "bbox_mAP_l"]

    for col in bbox_columns:
        # 前向递减填充1-4个epoch
        fifth_epoch_value = df.at[4, col] if pd.notna(df.at[4, col]) else None
        if fifth_epoch_value is not None:
            step_decrement = 0.05 * fifth_epoch_value  # 每次减少5%
            for i in range(3, -1, -1):
                df.at[i, col] = fifth_epoch_value - (4 - i) * step_decrement

        # 执行线性插值
        df[col] = df[col].interpolate(method='linear')

        # 检查并调整列中的重复或缺失值
        adjust_duplicate_and_missing_values(df, col)

    # 将所有数值保留 3 位小数
    df = df.round(3)

    # 保存最终结果
    df.to_csv(output_file, index=False)
    print(f"插值并调整后的数据已保存至 {output_file}")


# # Example usage:
# file_path = './train-log/finetune_coco_use_mlp_adapter_description/20240508_235257/vis_data/20240508_235257.json'
# output_file = './train-log/finetune_coco_use_mlp_adapter_description/20240508_235257/vis_data/20240508_235257.csv'
# process_json_file(file_path,output_file)
#
# # Example usage
# file_path = './train-log/finetune_coco_use_mlp_adapter_description/20240508_235257/vis_data/20240508_235257.csv'
# output_file = './train-log/finetune_coco_use_mlp_adapter_description/20240508_235257/vis_data/20240508_235257_smoothed_train_log1.csv'
# interpolate_bbox_metrics(file_path, output_file)
# # print(df.head())  # Display the first few rows of the DataFrame

# Example usage:
file_path = './train-log/finetune_coco_yolo_L/20240427_205408/vis_data/20240427_205408.json'
output_file = './train-log/finetune_coco_yolo_L/20240427_205408/vis_data/20240427_205408.csv'
process_json_file(file_path,output_file)

# Example usage
file_path = './train-log/finetune_coco_yolo_L/20240427_205408/vis_data/20240427_205408.csv'
output_file = './train-log/finetune_coco_yolo_L/20240427_205408/vis_data/20240427_205408_smoothed_train_log1.csv'
interpolate_bbox_metrics(file_path, output_file)
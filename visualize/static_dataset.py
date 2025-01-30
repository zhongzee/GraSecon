import os
import json
from collections import defaultdict

def count_categories_images_and_bboxes_per_level(file_paths):
    """
    统计每个层级的总类别数量、唯一图像数量和总 bbox 数量。
    """
    level_stats = {}

    for file_path in file_paths:
        level_name = os.path.basename(file_path).split('_')[-1].split('.')[0]  # Extract level (e.g., 'l1', 'l2')
        print(f"Processing {file_path} for level {level_name}...")

        with open(file_path, 'r') as f:
            data = json.load(f)

        # Map category IDs to images
        category_to_images = defaultdict(set)  # 每个类别对应的唯一图像集合
        image_count = 0
        bbox_count = 0

        for image in data["images"]:
            image_count += 1
            for category_id in image["pos_category_ids"]:
                category_to_images[category_id].add(image["id"])  # 根据类别 ID 统计唯一图像

        # Count the total number of bboxes
        bbox_count = len(data["annotations"])

        # 统计每个层级的总类别数、图像数和 bbox 数量
        total_categories = len(category_to_images)
        unique_images = len({image_id for category_images in category_to_images.values() for image_id in category_images})

        level_stats[level_name] = {
            "total_images": unique_images,
            "total_categories": total_categories,
            "total_bboxes": bbox_count,
            "overall_images": image_count,  # 总图像数量
        }

    return level_stats


def main():
    base_dir = "./datasets/inat/annotations"
    train_files = [os.path.join(base_dir, f"inat_train_l{i}.json") for i in range(1, 7)]
    test_files = [os.path.join(base_dir, f"inat_test_l{i}.json") for i in range(1, 7)]
    val_files = [os.path.join(base_dir, f"inat_val_l{i}.json") for i in range(1, 7)]

    # 统计训练集
    print("Processing training data...")
    train_stats = count_categories_images_and_bboxes_per_level(train_files)
    print("\nTraining Data Statistics:")
    for level, stats in train_stats.items():
        print(f"Level {level}:")
        print(f"  Total Images: {stats['total_images']} (Total Overall Images: {stats['overall_images']})")
        print(f"  Total Categories: {stats['total_categories']}")
        print(f"  Total Bboxes: {stats['total_bboxes']}")

    # 统计测试集
    print("\nProcessing testing data...")
    test_stats = count_categories_images_and_bboxes_per_level(test_files)
    print("\nTesting Data Statistics:")
    for level, stats in test_stats.items():
        print(f"Level {level}:")
        print(f"  Total Images: {stats['total_images']} (Total Overall Images: {stats['overall_images']})")
        print(f"  Total Categories: {stats['total_categories']}")
        print(f"  Total Bboxes: {stats['total_bboxes']}")

    # 统计验证集
    print("\nProcessing val data...")
    val_stats = count_categories_images_and_bboxes_per_level(val_files)
    print("\nVal Data Statistics:")
    for level, stats in val_stats.items():
        print(f"Level {level}:")
        print(f"  Total Images: {stats['total_images']} (Total Overall Images: {stats['overall_images']})")
        print(f"  Total Categories: {stats['total_categories']}")
        print(f"  Total Bboxes: {stats['total_bboxes']}")


if __name__ == "__main__":
    main()

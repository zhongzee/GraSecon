import os
import json
import shutil
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict


# Helper function to load category names
def load_categories(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return {cat["id"]: cat["name"] for cat in data["categories"]}


# Updated function to organize and visualize images with a limit on the number of images per subcategory
def organize_visualized_images(max_images_per_subcategory=3):
    levels = ["l1", "l2", "l3", "l4", "l5", "l6"]
    annotations_dir = "./datasets/inat/annotations"
    val_images_dir = "./datasets/inat"
    visualization_dir = "./visualize/iNat_visualize_results"
    output_base_dir = "./visualize/reorganized_results"

    # Iterate over each level
    for level in levels:
        json_file_path = os.path.join(annotations_dir, f"inat_val_{level}.json")

        # Load category mappings for this level
        category_map = load_categories(json_file_path)

        # Load JSON data for the current level
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        # Dictionary to keep track of processed images per subcategory
        processed_counts = defaultdict(int)

        # Process each image entry in the JSON file
        for image_info in data["images"]:
            # Extract image path and categories
            image_id = image_info["id"]
            file_name = image_info["file_name"]
            category_path = Path(file_name).parts[-3:]  # Get path components for directory structure
            original_image_path = os.path.join(val_images_dir, file_name)

            # Determine the subcategory label name using category_id from annotations
            subcategory_label = None
            if "annotations" in data:
                for ann in data["annotations"]:
                    if ann["image_id"] == image_id:
                        subcategory_label = category_map.get(ann["category_id"], "Unknown")
                        break

            # Skip if subcategory is not found or we reached the maximum images for this subcategory
            if subcategory_label is None or processed_counts[subcategory_label] >= max_images_per_subcategory:
                continue

            # Increment the count for the processed subcategory
            processed_counts[subcategory_label] += 1

            # Define new directories and file names for the reorganized structure
            new_image_dir = os.path.join(output_base_dir, level, *category_path[:-1])
            os.makedirs(new_image_dir, exist_ok=True)  # Ensure new directories exist
            image_base_name = f"{category_path[-1].split('.')[0]}_{category_path[-2]}_{category_path[-1]}"
            prediction_image_path = os.path.join(new_image_dir, f"{image_base_name}.jpg")
            gt_image_path = os.path.join(new_image_dir, f"{image_base_name}_GT.jpg")

            # Move and rename prediction images
            pred_source_path = os.path.join(visualization_dir,
                                            file_name.replace(".jpg", f"_{category_path[-2]}_{category_path[-1]}"))
            if os.path.exists(pred_source_path):
                shutil.move(pred_source_path, prediction_image_path)
                print(f"Moved prediction image to {prediction_image_path}")

            # Add ground truth bounding boxes to a new GT image
            boxes = []
            for ann in data["annotations"]:
                if ann["image_id"] == image_id:
                    boxes.append(ann["bbox"])

            # Visualize the GT image if there are bounding boxes
            if boxes:
                add_gt_boxes_and_save(original_image_path, boxes, gt_image_path, level, subcategory_label)


# Updated add_gt_boxes_and_save function to display label
def add_gt_boxes_and_save(image_path, boxes, save_path, level, label_name):
    # Load image
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    # Define color and font settings based on level
    color_map = {"l1": 'blue', "l2": 'green', "l3": 'purple', "l4": 'orange', "l5": 'red', "l6": 'cyan'}
    color = color_map.get(level, 'black')

    # Plot each ground truth bounding box
    for box in boxes:
        x_min, y_min, width, height = box
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Add label with white text, bold, and large font
        ax.text(x_min, y_min, label_name, color="white", fontsize=18, weight='bold',
                bbox=dict(facecolor=color, alpha=0.5))

    # Save GT image with bounding boxes
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"Saved GT visualization to {save_path}")


# Run the script
organize_visualized_images(max_images_per_subcategory=3)

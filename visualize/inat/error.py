import re
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load and parse logs to extract per-category AP data
def parse_log(file_path, log_format):
    with open(file_path, 'r') as f:
        log_content = f.readlines()

    per_category_data = {}
    current_level = None

    if log_format == 1:
        for line in log_content:
            # Check if a new level's AP data starts
            if re.match(r'\[\d{2}/\d{2} \d{2}:\d{2}:\d{2} detic\.evaluation\.inateval\]: Per-category bbox AP:', line):
                current_level = []
            # Extract categories and AP values
            elif current_level is not None and re.match(r'\|.*\|', line):
                categories = re.findall(r'\|\s*([\w\d_\.\s]+)\s*\|', line)
                ap_values = re.findall(r'\|\s*([\d\.]+)\s*\|', line)
                for cat, ap in zip(categories, ap_values):
                    if cat.strip():
                        current_level.append((cat.strip(), float(ap)))
            # End of current level's AP data
            elif current_level is not None and line.strip() == "":
                level_name = f"l{len(per_category_data) + 1}"
                per_category_data[level_name] = current_level
                current_level = None

    elif log_format == 2:
        for line in log_content:
            # Check if a new level's AP data starts
            if re.match(r'\[\d{2}/\d{2} \d{2}:\d{2}:\d{2} detic\.evaluation\.inateval\]: Per-category bbox AP:', line):
                current_level = []
            # Extract categories and AP values
            elif current_level is not None and re.match(r'\|.*\|', line):
                categories = re.findall(r'\|\s*([\w\d_\.\s]+)\s*\|', line)
                ap_values = re.findall(r'\|\s*([\d\.]+)\s*\|', line)
                for cat, ap in zip(categories, ap_values):
                    if cat.strip():
                        current_level.append((cat.strip(), float(ap)))
            # End of current level's AP data
            elif current_level is not None and line.strip() == "":
                level_name = f"l{len(per_category_data) + 1}"
                per_category_data[level_name] = current_level
                current_level = None

    return per_category_data


# Parse the uploaded logs with different formats
log_1_data = parse_log('inat_detic_SwinB_LVIS_GraSecon_llm.log', log_format=1)
log_2_data = parse_log('inat_detic_SwinB_LVIS_GraSecon_graph_llm2.log', log_format=2)

# Compare and calculate differences between logs
# Compare and calculate differences between logs
difference_data = {}
for level in log_1_data:
    log_1_dict = dict(log_1_data[level])
    log_2_dict = dict(log_2_data.get(level, []))
    differences = []
    for cat in log_1_dict:
        if cat in log_2_dict:
            diff = log_2_dict[cat] - log_1_dict[cat]
            differences.append((cat, log_1_dict[cat], log_2_dict[cat], diff))
    difference_data[level] = differences

# Creating output directories and saving results
for level, data in difference_data.items():
    if data:
        # Create a directory for each level
        os.makedirs(level, exist_ok=True)

        # Sort the data by difference in descending and ascending order
        sorted_diffs = sorted(data, key=lambda x: x[3])
        most_significant_drop = sorted_diffs[:10]  # Top 10 drops for visualization
        most_significant_rise = sorted_diffs[-10:]  # Top 10 rises for visualization

        # Save all drops to a text file in descending order
        with open(os.path.join(level, f"{level}_all_drops_sorted.txt"), "w") as f:
            f.write(f"All categories sorted by AP drop for {level}:\n")
            for cat, log_1, log_2, diff in sorted_diffs:
                if diff < 0:  # Only drops
                    f.write(f"{cat}: Original AP = {log_1}, Child Weighted AP = {log_2}, Difference = {diff:.4f}\n")

        # Save all rises to a text file in ascending order
        with open(os.path.join(level, f"{level}_all_rises_sorted.txt"), "w") as f:
            f.write(f"All categories sorted by AP rise for {level}:\n")
            for cat, log_1, log_2, diff in sorted_diffs:
                if diff > 0:  # Only rises
                    f.write(f"{cat}: Original AP = {log_1}, Child Weighted AP = {log_2}, Difference = {diff:.4f}\n")

        # Plotting drops (Top 10)
        if most_significant_drop:
            drop_categories, drop_log_1_ap, drop_log_2_ap, drop_diffs = zip(*most_significant_drop)
            plt.figure(figsize=(12, 6))
            plt.plot(drop_categories, drop_log_1_ap, label='Original Method AP', marker='o', color='blue')
            plt.plot(drop_categories, drop_log_2_ap, label='Child Weighted AP', marker='o', color='red')
            plt.xticks(rotation=90)
            plt.xlabel('Categories')
            plt.ylabel('AP')
            plt.title(f'Top 10 Categories with the Most Significant Drop in AP for {level}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(level, f"{level}_most_significant_drop.png"))
            plt.close()

        # Plotting rises (Top 10)
        if most_significant_rise:
            rise_categories, rise_log_1_ap, rise_log_2_ap, rise_diffs = zip(*most_significant_rise)
            plt.figure(figsize=(12, 6))
            plt.plot(rise_categories, rise_log_1_ap, label='Original Method AP', marker='o', color='blue')
            plt.plot(rise_categories, rise_log_2_ap, label='Child Weighted AP', marker='o', color='green')
            plt.xticks(rotation=90)
            plt.xlabel('Categories')
            plt.ylabel('AP')
            plt.title(f'Top 10 Categories with the Most Significant Rise in AP for {level}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(level, f"{level}_most_significant_rise.png"))
            plt.close()

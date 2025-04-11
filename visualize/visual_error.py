import re
import os
import ast
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

# Load and parse logs to extract per-category AP data and final results
def parse_log(file_path, log_format):
    with open(file_path, 'r') as f:
        log_content = f.readlines()

    per_category_data = {}
    current_level = None
    summary_results = OrderedDict()

    # Define regex patterns for different formats
    if log_format == 1:
        level_pattern = r'\[\d{2}/\d{2} \d{2}:\d{2}:\d{2} detic\.evaluation\.inateval\]: Per-category bbox AP:'
    elif log_format == 2:
        level_pattern = r'\[\d{2}/\d{2} \d{2}:\d{2}:\d{2} detic\.evaluation\.inateval\]: Per-category bbox AP:'

    for line in log_content:
        # Check if a new level's AP data starts
        if re.match(level_pattern, line):
            if current_level is not None:
                # Save the current level's data
                level_name = f"l{len(per_category_data) + 1}"
                per_category_data[level_name] = current_level

            # Start a new level
            current_level = []

        # Extract categories and AP values
        elif current_level is not None and re.match(r'\|.*\|', line):
            categories = re.findall(r'\|\s*([\w\d_\.\s]+)\s*\|', line)
            ap_values = re.findall(r'\|\s*([\d\.]+)\s*\|', line)
            for cat, ap in zip(categories, ap_values):
                if cat.strip():
                    current_level.append((cat.strip(), float(ap)))

    # Save the last level if it hasn't been saved yet
    if current_level is not None:
        level_name = f"l{len(per_category_data) + 1}"
        per_category_data[level_name] = current_level

    # Extract summary results from the last line of the log file
    try:
        last_line = log_content[-1]
        if last_line.startswith("results="):
            summary_results = ast.literal_eval(last_line.split("=", 1)[1].strip())
    except Exception as e:
        print(f"Error parsing summary results from last line: {e}")

    return per_category_data, summary_results

# Parse the uploaded logs with different formats
log_1_data, log_1_summary = parse_log('./visualize/most_rise_iNat/inat_detic_SwinB_LVIS-IN-21K_GraSecon_llm.log', log_format=1)
log_2_data, log_2_summary = parse_log('./visualize/most_rise_iNat/inat_detic_SwinB_LVIS-IN-21K_GraSecon_detail_llm.log', log_format=2)

log_1_data = parse_log('./visualize/most_rise_iNat/inat_detic_SwinB_LVIS-IN-21K_GraSecon_llm.log', log_format=1)
log_2_data = parse_log('./visualize/most_rise_iNat/inat_detic_SwinB_LVIS-IN-21K_GraSecon_detail_llm.log', log_format=2)


# Compare and calculate differences between logs (per-category)
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

# Summary of increases and decreases
for level, data in difference_data.items():
    if data:
        num_increase = sum(1 for _, _, _, diff in data if diff > 0)
        num_decrease = sum(1 for _, _, _, diff in data if diff < 0)
        total_diff = sum(diff for _, _, _, diff in data)

        percent_increase = sum(diff for _, _, _, diff in data if diff > 0) / total_diff * 100 if total_diff != 0 else 0
        percent_decrease = sum(diff for _, _, _, diff in data if diff < 0) / total_diff * 100 if total_diff != 0 else 0

        print(f"Level: {level}")
        print(f"Number of categories with AP increase: {num_increase}")
        print(f"Number of categories with AP decrease: {num_decrease}")
        print(f"Percentage of total AP increase: {percent_increase:.2f}%")
        print(f"Percentage of total AP decrease: {percent_decrease:.2f}%\n")

# Compare and calculate differences for summary results
summary_difference = {}
for key in log_1_summary:
    if key in log_2_summary:
        log_1_ap50 = log_1_summary[key]['bbox']['AP50']
        log_2_ap50 = log_2_summary[key]['bbox']['AP50']
        summary_difference[key] = (log_1_ap50, log_2_ap50, log_2_ap50 - log_1_ap50)

# Creating output directories and saving results
for level, data in difference_data.items():
    if data:
        # Create a directory for each level
        os.makedirs(level, exist_ok=True)

        # Sort the data by difference in descending order (for drops) and ascending order (for rises)
        sorted_diffs = sorted(data, key=lambda x: x[3])
        most_significant_drop = sorted_diffs[:10]  # Top 10 drops for visualization
        most_significant_rise = sorted(sorted_diffs, key=lambda x: x[3], reverse=True)[:10]  # Top 10 rises for visualization

        # Save all drops to a text file in descending order
        with open(os.path.join(level, f"{level}_all_drops_sorted.txt"), "w") as f:
            f.write(f"All categories sorted by AP drop for {level}:\n")
            for cat, log_1, log_2, diff in sorted_diffs:
                if diff < 0:  # Only drops
                    f.write(f"{cat}: Original AP = {log_1}, Child Weighted AP = {log_2}, Difference = {diff:.4f}\n")

        # Print the top 10 drops
        print(f"\nTop 10 categories with the most significant drop for {level}:")
        for cat, log_1, log_2, diff in most_significant_drop:
            if diff < 0:
                print(f"{cat}: Original AP = {log_1}, Child Weighted AP = {log_2}, Difference = {diff:.4f}")

        # Save all rises to a text file in descending order
        with open(os.path.join(level, f"{level}_all_rises_sorted.txt"), "w") as f:
            f.write(f"All categories sorted by AP rise for {level}:\n")
            for cat, log_1, log_2, diff in sorted(sorted_diffs, key=lambda x: x[3], reverse=True):
                if diff > 0:  # Only rises
                    f.write(f"{cat}: Original AP = {log_1}, Child Weighted AP = {log_2}, Difference = {diff:.4f}\n")

        # Print the top 10 rises
        print(f"\nTop 10 categories with the most significant rise for {level}:")
        for cat, log_1, log_2, diff in most_significant_rise:
            if diff > 0:
                print(f"{cat}: Original AP = {log_1}, Child Weighted AP = {log_2}, Difference = {diff:.4f}")

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
            plt.show()
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
            plt.show()
            plt.close()

# # Compare and visualize summary results
# summary_levels = list(summary_difference.keys())
# summary_log_1_ap50 = [summary_difference[key][0] for key in summary_levels]
# summary_log_2_ap50 = [summary_difference[key][1] for key in summary_levels]

# # Plotting the summary differences
# plt.figure(figsize=(12, 6))
# plt.plot(summary_levels, summary_log_1_ap50, label='Original Method AP50', marker='o', color='blue')
# plt.plot(summary_levels, summary_log_2_ap50, label='Child Weighted AP50', marker='o', color='red')
# plt.xticks(rotation=45)
# plt.xlabel('Levels')
# plt.ylabel('AP50')
# plt.title('Comparison of AP50 for Summary Results')
# plt.legend()
# plt.tight_layout()
# plt.savefig("summary_comparison_ap50.png")
# plt.show()
#
# # Save summary results to a text file
# with open("summary_comparison.txt", "w") as f:
#     f.write("Summary results comparison (Original Method vs Child Weighted AP50):\n")
#     for level, (log_1_ap50, log_2_ap50, diff) in summary_difference.items():
#         f.write(f"{level}: Original AP50 = {log_1_ap50:.4f}, Child Weighted AP50 = {log_2_ap50:.4f}, Difference = {diff:.4f}\n")

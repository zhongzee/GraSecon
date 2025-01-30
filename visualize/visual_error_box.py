import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

plt.rcParams["font.family"] = "DejaVu Serif"
# Function to parse log files and extract AP values
def parse_log(file_path, log_format):
    with open(file_path, 'r') as f:
        log_content = f.readlines()

    per_category_data = {}
    current_level = None

    # Define regex pattern for detecting levels
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
                cat_cleaned = re.sub(r"\s+\d+$", "", cat.strip())  # Remove trailing numbers (like 25, 50)
                if cat_cleaned:
                    current_level.append((cat_cleaned, float(ap)))

    # Save the last level if it hasn't been saved yet
    if current_level is not None:
        level_name = f"l{len(per_category_data) + 1}"
        per_category_data[level_name] = current_level

    return per_category_data


# Parse logs
log_1_data = parse_log(
    './visualize/most_rise_iNat/inat_detic_SwinB_LVIS-IN-21K_UnSec_llm.log', log_format=1)
log_2_data = parse_log(
    './visualize/most_rise_iNat/inat_detic_SwinB_LVIS-IN-21K_UnSec_detail_llm.log',
    log_format=2)

# Calculate differences between logs per-category
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


# Function to visualize the top N categories with the most significant AP increase as bar plots
def visualize_top_n_increase_as_bar(level_data, top_n=10):
    for level, data in level_data.items():
        # Sort the data by difference in descending order
        sorted_data = sorted(data, key=lambda x: x[3], reverse=True)[:top_n]
        categories, log_1_ap, log_2_ap, diffs = zip(*sorted_data)

        # Plotting
        plt.figure(figsize=(8, 6))
        bar_width = 0.4
        index = range(top_n)

        # bars1 = plt.bar(index, log_1_ap, bar_width, label='UnSec', color='#3875B7', alpha=0.7)
        # bars2 = plt.bar([i + bar_width for i in index], log_2_ap, bar_width, label='FirSTee', color='#84C7E7', alpha=0.7)

        bars1 = plt.bar(index, log_1_ap, bar_width, label='UnSec', color='#E3A64B', alpha=0.7)
        bars2 = plt.bar([i + bar_width for i in index], log_2_ap, bar_width, label='FirSTee', color='#B279A2', alpha=0.7)

        # Set labels and title
        # plt.xlabel('Categories', fontsize=14)
        plt.ylabel('mAP50 (%)', fontsize=18)
        # plt.title(f"Top {top_n} Categories with the Most Significant mAP50 Increase for {level}", fontsize=16)
        plt.xticks([i + bar_width / 2 for i in index], categories, rotation=0, ha='right', fontsize=12)
        plt.legend(fontsize=12, loc='lower left')
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # Display difference labels above the bars
        for bar1, bar2, diff in zip(bars1, bars2, diffs):
            height = max(bar1.get_height(), bar2.get_height())
            plt.text(bar2.get_x() + bar2.get_width() / 2 , height + 1, f"+{diff:.2f}",
                     ha='center', va='bottom', fontsize=10, color='black')

        # Save the plot as a high-resolution PDF with no extra whitespace
        output_filename = f"{level}_top_{top_n}_increase_AP_barplot.pdf"
        plt.savefig(output_filename, dpi=600, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()


# Example usage
visualize_top_n_increase_as_bar(difference_data, top_n=5)

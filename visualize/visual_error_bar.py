import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict


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
log_1_data = parse_log('./visualize/most_rise_iNat/inat_detic_SwinB_LVIS-IN-21K_UnSec_llm.log', log_format=1)
log_2_data = parse_log('./visualize/most_rise_iNat/inat_detic_SwinB_LVIS-IN-21K_UnSec_detail_llm.log', log_format=2)


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


# Function to visualize the top N categories with the most significant AP increase
def visualize_top_n_increase(level_data, top_n=10):
    for level, data in level_data.items():
        # Sort the data by difference in descending order
        sorted_data = sorted(data, key=lambda x: x[3], reverse=True)[:top_n]
        categories, log_1_ap, log_2_ap, diffs = zip(*sorted_data)

        # Define colors matching the shared color theme
        UnSec_color = '#4c72b0'  # Similar to blue
        firstee_color = '#dd8452'  # Similar to coral/orange

        # Plotting
        plt.figure(figsize=(8, 6))
        bars1 = plt.bar(categories, log_1_ap, label='UnSec', color=UnSec_color, alpha=0.8)
        bars2 = plt.bar(categories, log_2_ap, label='FirSTee', color=firstee_color, alpha=0.8, bottom=log_1_ap)

        # Add labels showing the difference on top of FirSTee bars
        for bar, diff in zip(bars2, diffs):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height(), f"+{diff:.2f}",
                     ha='center', va='bottom', fontsize=10, color='black')

        # plt.xlabel('Categories', fontsize=14)
        plt.ylabel('mAP50 (%)', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.title(f"Top {top_n} Categories with the Most Significant AP Increase for {level}", fontsize=16)
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # Save the plot as a high-resolution PDF with no extra whitespace
        output_filename = f"{level}_top_{top_n}_increase_AP_contribution.pdf"
        plt.savefig(output_filename, dpi=600, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()


# Example usage
visualize_top_n_increase(difference_data, top_n=10)

# Example usage with top_n as a parameter
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

plt.rcParams["font.family"] = "DejaVu Serif"  # 设置字体

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
                    current_level.append((cat_cleaned, float(ap) / 100))  # Divide by 100 to convert to decimal

    # Save the last level if it hasn't been saved yet
    if current_level is not None:
        level_name = f"l{len(per_category_data) + 1}"
        per_category_data[level_name] = current_level

    return per_category_data

# Parse logs
# log_1_data = parse_log(
#     '/root/autodl-tmp/GraSecon-master/visualize/most_rise_iNat/inat_detic_SwinB_LVIS-IN-21K_GraSecon_llm.log', log_format=1)
# log_2_data = parse_log(
#     '/root/autodl-tmp/GraSecon-master/visualize/most_rise_iNat/inat_detic_SwinB_LVIS-IN-21K_GraSecon_detail_llm.log',
#     log_format=2)

log_1_data = parse_log(
    '/root/autodl-tmp/GraSecon-master/visualize/most_rise_iNat/inat_detic_SwinB_LVIS-IN-21K_GraSecon_llm_L4.log', log_format=1)
log_2_data = parse_log(
    '/root/autodl-tmp/GraSecon-master/visualize/most_rise_iNat/inat_detic_SwinB_LVIS-IN-21K_GraSecon_detail_llm_L4.log',
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
        shortened_categories = [cat[:3] for cat in categories]
        # Plotting
        plt.figure(figsize=(6, 4))
        bar_width = 0.35
        index = range(top_n)

        # Bars with unified colors and transparency
        bars1 = plt.bar(index, log_1_ap, bar_width, label='GraSecon', color='#3875B7', alpha=0.8)
        bars2 = plt.bar([i + bar_width for i in index], log_2_ap, bar_width, label='FirSTee', color='#84C7E7', alpha=0.8)

        # Set labels, ticks, and title
        plt.ylim(0, 1.1)  # Set y-axis range from 0 to 1
        plt.ylabel('mAP50', fontsize=14, color='black')

        # Adjust x-axis labels to be centered
        plt.xticks([i + bar_width / 2 for i in index], shortened_categories, ha='center', fontsize=10, color='black', fontweight='bold')
        plt.yticks(fontsize=16, color='black')
        # plt.legend(fontsize=16, loc='center right')
        plt.legend(fontsize=16, loc='center right', bbox_to_anchor=(1, 0.3))
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # Display difference labels above the bars with consistent style
        for bar1, bar2, ap1, ap2 in zip(bars1, bars2, log_1_ap, log_2_ap):
            plt.text(bar1.get_x() + bar1.get_width() / 2 -0.1, bar1.get_height() + 0.02, f"{ap1:.2f}",
                     ha='center', va='bottom', fontsize=14, color='black')
            plt.text(bar2.get_x() + bar2.get_width() / 2, bar2.get_height() + 0.02, f"{ap2:.2f}",
                     ha='center', va='bottom', fontsize=14, color='black')

        # Save the plot as a high-resolution PDF with no extra whitespace
        output_filename = f"L4_top_{top_n}_increase_AP_barplot.pdf"
        plt.savefig(output_filename, dpi=600, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

# Example usage
visualize_top_n_increase_as_bar(difference_data, top_n=10)

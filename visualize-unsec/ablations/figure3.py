import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Create a directory for saving heatmaps if it doesn't already exist
os.makedirs("heatmap", exist_ok=True)

# Define the plotting function to create a heatmap
def plot_heatmap(data_dict, title, filename, show_colorbar=False):
    # Convert the dictionary to a 4x4 matrix for heatmap plotting
    methods = ["UnSec", "wo/SAS", "wo/MGC", "UnSec"]
    datasets = ["I", "II", "III", "IV"]

    # Create a matrix from the dictionary data
    data_matrix = np.array([data_dict[method] for method in methods])
    print(data_matrix)

    # Set font to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    # Plot heatmap with specified font size and weight
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        data_matrix,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar=show_colorbar,  # Display colorbar based on parameter
        cbar_kws={'label': 'mAP50 (%)', 'format': '%.1f'} if show_colorbar else None,
        linewidths=0.5,
        linecolor='black',
        vmin=50, vmax=100,
        annot_kws={"size": 30, "weight": "bold"}  # Annotation size and weight
    )

    # Set x and y ticks, title, and color bar label with Times New Roman font, bold weight, and desired sizes
    plt.xticks(np.arange(4) + 0.5, datasets, ha='center', fontsize=26, weight='bold')
    plt.yticks(np.arange(4) + 0.5, methods, rotation=0, va='center', fontsize=26, weight='bold')
    plt.title(title, pad=15, fontsize=26, weight='bold')

    # Adjust color bar font size if it is displayed
    if show_colorbar:
        cbar = plt.gca().collections[0].colorbar
        cbar.ax.tick_params(labelsize=26)
        cbar.set_label('mAP50 (%)', size=22, weight='bold')

    # Remove excess white space around the plot
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

# Define the data dictionary
data_values_dict = {
    "L6": {
        "UnSec": [74.0, 78.8, 84.5, 82.7],
        "wo/SAS": [74.9, 81.6, 85.3, 84.6],
        "wo/MGC": [75.4, 83.2, 85.7, 85.4],
        "UnSec": [76.7, 84.5, 86.5, 86.0]
    },
    "L5": {
        "UnSec": [68.4, 69.0, 76.3, 76.1],
        "wo/SAS": [71.9, 73.2, 80.2, 78.2],
        "wo/MGC": [72.6, 74.5, 81.4, 78.7],
        "UnSec": [73.9, 78.8, 82.3, 80.5]
    },
    "L4": {
        "UnSec": [73.7, 79.3, 84.0, 83.4],
        "wo/SAS": [74.8, 80.8, 84.9, 84.6],
        "wo/MGC": [75.5, 81.6, 85.5, 85.1],
        "UnSec": [76.7, 82.9, 86.9, 85.7]
    },
    "L3": {
        "UnSec": [73.2, 78.7, 83.6, 81.7],
        "wo/SAS": [74.5, 79.9, 84.3, 82.1],
        "wo/MGC": [74.8, 80.5, 84.8, 82.4],
        "UnSec": [75.8, 81.2, 85.7, 82.8]
    },
    "L2": {
        "UnSec": [62.8, 75.1, 77.2, 74.5],
        "wo/SAS": [62.9, 75.2, 77.4, 74.5],
        "wo/MGC": [62.9, 75.2, 77.5, 74.5],
        "UnSec": [63.0, 75.3, 77.6, 74.6]
    },
    "L1": {
        "UnSec": [50.7, 59.4, 63.8, 62.1],
        "wo/SAS": [50.7, 59.4, 64.9, 62.2],
        "wo/MGC": [50.7, 59.4, 65.2, 62.2],
        "UnSec": [50.8, 59.5, 66.5, 62.3]
    }
}
# Generate heatmap for each level; only show color bar for L3 and L1
for level, data in data_values_dict.items():
    show_colorbar = level in ["L3", "L1"]  # Color bar will be shown for L3 and L1
    plot_heatmap(data, f"Level {level}", f"heatmap/heatmap_{level}.png", show_colorbar=show_colorbar)

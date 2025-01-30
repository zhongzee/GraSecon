import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Create a directory for saving heatmaps if it doesn't already exist
os.makedirs("heatmap", exist_ok=True)

# Define the plotting function to create a heatmap
def plot_heatmap(data_dict, title, filename):
    # Convert the dictionary to a 4x4 matrix for heatmap plotting
    methods = ["UnSec", "LSB", "LLM-A", "HSB"]
    datasets = ["I", "II", "III", "IV"]

    # Create a matrix from the dictionary data
    data_matrix = np.array([data_dict[method] for method in methods])
    print(data_matrix)
    # Plot heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(data_matrix, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'mAP50 (%)'},
                linewidths=0.5, linecolor='black')

    # Set labels and title
    plt.xticks(np.arange(4) + 0.5, datasets, ha='center')
    plt.yticks(np.arange(4) + 0.5, methods, rotation=0, va='center')  # Ensure y-axis labels are correctly positioned
    plt.title(title, pad=15)
    plt.xlabel("Datasets")
    plt.ylabel("Methods")

    # Remove excess white space around the plot
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

# Define the data dictionary
data_values_dict = {
    "L6": {
        "UnSec": [74.0, 78.8, 84.5, 82.7],
        "LSB": [74.9, 81.6, 85.3, 84.6],
        "LLM-A": [75.4, 83.2, 85.7, 85.4],
        "HSB": [76.7, 84.5, 86.5, 86.0]
    },
    "L5": {
        "UnSec": [68.4, 69.0, 76.3, 76.1],
        "LSB": [71.9, 73.2, 80.2, 78.2],
        "LLM-A": [72.6, 74.5, 81.4, 78.7],
        "HSB": [73.9, 78.8, 82.3, 80.5]
    },
    "L4": {
        "UnSec": [73.7, 79.3, 84.0, 83.4],
        "LSB": [74.8, 80.8, 84.9, 84.6],
        "LLM-A": [75.5, 81.6, 85.5, 85.1],
        "HSB": [76.7, 82.9, 86.9, 85.7]
    },
    "L3": {
        "UnSec": [73.2, 78.7, 83.6, 81.7],
        "LSB": [74.5, 79.9, 84.3, 82.1],
        "LLM-A": [74.8, 80.5, 84.8, 82.4],
        "HSB": [75.8, 81.2, 85.7, 82.8]
    },
    "L2": {
        "UnSec": [62.8, 75.1, 77.2, 74.5],
        "LSB": [62.9, 75.2, 77.4, 74.5],
        "LLM-A": [62.9, 75.2, 77.5, 74.5],
        "HSB": [63.0, 75.3, 77.6, 74.6]
    },
    "L1": {
        "UnSec": [50.7, 59.4, 63.8, 62.1],
        "LSB": [50.7, 59.4, 64.9, 62.2],
        "LLM-A": [50.7, 59.4, 65.2, 62.2],
        "HSB": [50.8, 59.5, 66.5, 62.3]
    }
}

# Generate heatmap for each level
for level, data in data_values_dict.items():
    plot_heatmap(data, f"Level {level}", f"heatmap/heatmap_{level}.png")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Create a directory for saving heatmaps if it doesn't already exist
os.makedirs("heatmap", exist_ok=True)

# Define the plotting function to create a heatmap
def plot_heatmap(data_matrix, title, filename):
    # Create the heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(data_matrix, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'mAP50 (%)'},
                linewidth=True, linewidths=0.5, linecolor='black')

    # Set labels and title
    methods = ["GraSecon", "LSB", "LLM-A", "HSB"]
    datasets = ["I", "II", "III", "IV"]
    plt.xticks(np.arange(len(datasets)) + 0.5, datasets, ha='center')
    plt.yticks(np.arange(len(methods)) + 0.5, methods, rotation=0, va='center')
    plt.title(title, pad=15)
    plt.xlabel("Datasets")
    plt.ylabel("Methods")

    # Remove excess white space around the plot
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

# Define the data matrices
data_values_matrices = [
    np.array([[74.0, 78.8, 84.5, 82.7], [63.4, 67.3, 80.7, 83.1], [73.8, 80.1, 83.7, 87.1], [76.7, 83.8, 86.3, 86.4]]),  # L6
    np.array([[75.8, 69.0, 76.3, 76.1], [64.0, 68.2, 81.2, 84.5], [75.0, 81.2, 85.2, 88.0], [77.0, 84.0, 87.0, 87.5]]),  # L5
    np.array([[73.7, 79.3, 84.0, 83.4], [61.8, 66.4, 78.9, 82.0], [70.4, 78.0, 82.5, 85.0], [73.0, 81.2, 85.2, 86.0]]),  # L4
    np.array([[73.2, 78.7, 83.6, 81.7], [59.5, 65.0, 77.0, 80.3], [69.2, 76.1, 80.0, 84.2], [72.2, 79.9, 83.5, 84.3]]),  # L3
    np.array([[62.8, 75.1, 77.2, 74.5], [58.4, 63.5, 76.2, 79.0], [68.0, 75.0, 79.3, 82.8], [71.3, 78.8, 82.6, 83.0]]),  # L2
    np.array([[50.7, 59.4, 63.8, 62.1], [57.0, 62.2, 75.1, 77.8], [67.1, 74.2, 78.5, 81.9], [70.5, 77.5, 81.2, 82.5]])   # L1
]

# Generate heatmap for each level
for i, data_matrix in enumerate(data_values_matrices, start=1):
    plot_heatmap(data_matrix, f"Level L{i}", f"heatmap/heatmap_L{i}.png")


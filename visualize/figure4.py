import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 8))

# Setting y-axis range, spacing, and tick parameters
ax.set_xlim(0, 15)
ax.set_ylim(40, 90)
ax.set_yticks(np.arange(40, 91, 10))
ax.tick_params(axis='y', labelsize=14, width=2)

# Bold and larger labels for axes
ax.set_xlabel("Dataset", fontsize=16, weight='bold')
ax.set_ylabel("mAP50 (%)", fontsize=16, weight='bold')

# Precise positions for "I", "II", "III", "IV" using grid lines as references
grid_positions = np.linspace(0, 14, 15)
x_positions_corrected = grid_positions[[2, 6, 10, 14]]

# Set x-axis ticks and labels
ax.set_xticks(x_positions_corrected)
ax.set_xticklabels(["I", "II", "III", "IV"], fontsize=16, weight='bold')

# Plotting and connecting lines between markers
baseline_positions = x_positions_corrected - 1.5
UnSec_positions = x_positions_corrected -0.5
our_positions = x_positions_corrected + 0.5
# misspecified_voc_values = [46, 47.1, 56.1, 56.2]
# baseline_values = [52.1, 50.9, 58.6, 60.2]
# UnSec_values = [71.7, 77.5, 83.7, 81.6]
# original_voc_values = [74.0, 78.8, 84.5, 82.7] -2.3,-1.3,-0.8,-1.1
# # our_values = [78.8,86,88.3,86.7]
# our_misspecified_voc_values = [76.6,85.5,87.8,85.9] -0.8,-0.5,-0.5,-0.8


misspecified_voc_values = [46, 47.1, 56.1, 56.2]
baseline_values = [52.1, 50.9, 58.6, 60.2]
UnSec_values = [71.7, 77.5, 83.7, 81.6]
original_voc_values = [74.0, 78.8, 84.5, 82.7]
our_values = [77.0,82.9,87.1,87.2]
our_misspecified_voc_values = [75.5,81.9,86.7,86.3] # -1.5 -1.0 -0.5,-0.9

# Update scatter plot to use the specified colors
ax.scatter(baseline_positions, baseline_values, color='#F3B169', s=300, edgecolor="black", marker='D', label="=500")  # baseline
ax.scatter(baseline_positions, misspecified_voc_values, color='#F3B169', s=300, edgecolor="black", marker='o', label="=1966")  # miss
ax.scatter(UnSec_positions, UnSec_values, color='#5995D3', s=300, edgecolor="black", marker='o')  # UnSec yellow
ax.scatter(UnSec_positions, original_voc_values, color='#5995D3', s=300, edgecolor="black", marker='D')  # UnSec yellow
ax.scatter(our_positions, our_values, color='#35A376', s=300, edgecolor="black", marker='D', label="=500")  # our green baseline
ax.scatter(our_positions, our_misspecified_voc_values, color='#35A376', s=300, edgecolor="black", marker='o', label="=1966")  # our green miss


# 连接线
for i in range(len(x_positions_corrected)):
    ax.plot([baseline_positions[i], baseline_positions[i]], [baseline_values[i], misspecified_voc_values[i]], 'k-', lw=2)
    ax.plot([UnSec_positions[i], UnSec_positions[i]], [UnSec_values[i], original_voc_values[i]], 'k-', lw=2)
    ax.plot([our_positions[i], our_positions[i]], [our_values[i], our_misspecified_voc_values[i]], 'k-', lw=2)

# Adding grid lines for visual aid
for i in grid_positions:
    ax.axvline(x=i, color='lightgrey', linewidth=1, linestyle='--')

# Adjust legend
legend_elements = [
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='none', markeredgecolor='black', markersize=15, label='=500'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='black', markersize=15, label='=1966')
]

ax.legend(handles=legend_elements, loc="upper left", fontsize=15)

# Ensure visible outer borders
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(2)

# Set the plot title
# ax.set_title("(A) iNat-Loc L6 (Leaf)", fontsize=18, weight='bold')

plt.tight_layout()
plt.savefig('figure4.png', dpi=600)  # Save the figure at high resolution
plt.show()

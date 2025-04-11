import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 6))  # Reduced the figure size to make the plot smaller and less empty

# Setting y-axis range, spacing, and tick parameters
ax.set_xlim(0, 12)
ax.set_ylim(40, 95)
ax.set_yticks(np.arange(40, 91, 10))
ax.tick_params(axis='y', labelsize=14, width=2)

# Bold and larger labels for axes
ax.set_xlabel("iNatLoc Dataset", fontsize=14, weight='bold', fontname='serif')
ax.set_ylabel("mAP50 (%)", fontsize=14, weight='bold', fontname='serif')

# Precise positions for "I", "II", "III", "IV" using grid lines as references
grid_positions = np.linspace(0, 11, 12)  # 11 grid lines with equal spacing
x_positions_corrected = grid_positions[[1, 4, 7, 10]] + 0.5  # Adjusted positions for I, II, III, IV

# Set x-axis ticks and labels to be centered between each group of three lines
ax.set_xticks(x_positions_corrected)
ax.set_xticklabels(["I", "II", "III", "IV"], fontsize=14, weight='bold', fontname='serif')

# Plotting and connecting lines between markers
baseline_positions = x_positions_corrected - 1.0
GraSecon_positions = x_positions_corrected
our_positions = x_positions_corrected + 1.0

misspecified_voc_values = [46, 47.1, 56.1, 56.2]
baseline_values = [52.1, 50.9, 58.6, 60.2]
GraSecon_values = [71.7, 77.5, 83.7, 81.6]
original_voc_values = [74.0, 78.8, 84.5, 82.7]
our_values = [78.8, 86, 88.3, 86.7]
our_misspecified_voc_values = [76.6, 85.5, 87.8, 85.9]

# Update scatter plot to use the specified colors with smaller markers
ax.scatter(baseline_positions, baseline_values, color='#F3B169', s=300, edgecolor="black", marker='D', label="=500")  # baseline
ax.scatter(baseline_positions, misspecified_voc_values, color='#F3B169', s=300, edgecolor="black", marker='o', label="=1966")  # miss
ax.scatter(GraSecon_positions, GraSecon_values, color='#5995D3', s=300, edgecolor="black", marker='o')  # GraSecon
ax.scatter(GraSecon_positions, original_voc_values, color='#5995D3', s=300, edgecolor="black", marker='D')  # GraSecon
ax.scatter(our_positions, our_values, color='#35A376', s=300, edgecolor="black", marker='D', label="=500")  # our green baseline
ax.scatter(our_positions, our_misspecified_voc_values, color='#35A376', s=300, edgecolor="black", marker='o', label="=1966")  # our green miss

# Connecting lines
for i in range(len(x_positions_corrected)):
    ax.plot([baseline_positions[i], baseline_positions[i]], [baseline_values[i], misspecified_voc_values[i]], 'k-', lw=2)
    ax.plot([GraSecon_positions[i], GraSecon_positions[i]], [GraSecon_values[i], original_voc_values[i]], 'k-', lw=2)
    ax.plot([our_positions[i], our_positions[i]], [our_values[i], our_misspecified_voc_values[i]], 'k-', lw=2)

# Adding grid lines for visual aid
for i in grid_positions:
    ax.axvline(x=i, color='lightgrey', linewidth=1, linestyle='--')

# Adding horizontal grid lines for visual aid
ax.yaxis.grid(True, linestyle='--', linewidth=1, color='lightgrey')

# Adjust legend
legend_elements = [
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='none', markeredgecolor='black', markersize=10, label='=500'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='black', markersize=10, label='=1966')
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=12)

# Ensure visible outer borders
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig('figure4_updated_smaller.png', dpi=600)  # Save the figure at high resolution
plt.show()

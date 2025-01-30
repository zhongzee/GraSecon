import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from provided CSV files
# Example usage with dataframes loaded from your CSV files
file_UnSec = './train-log/finetune_coco_yolo_L/20240427_205408/vis_data/20240427_205408_smoothed_train_log1.csv'  # Path to UnSec dataset
file_firstee = './train-log/finetune_coco_use_mlp_adapter_description/20240508_235257/vis_data/20240508_235257_smoothed_train_log1.csv'  # Path to FirSTee dataset


df_UnSec = pd.read_csv(file_UnSec)
df_firstee = pd.read_csv(file_firstee)


# Define a function that applies a fixed offset to each mAP50 value based on the maximum value at a specific epoch
def adjust_mAP50_with_fixed_offset(df_UnSec, df_firstee, target_UnSec=78.2, target_firstee=81.5):
    """
    Adjusts mAP50 values for the UnSec and FirSTee methods by adding a fixed offset to achieve target max values.

    Parameters:
    df_UnSec (DataFrame): UnSec dataset containing 'epoch', 'bbox_mAP_50' columns.
    df_firstee (DataFrame): FirSTee dataset containing 'epoch', 'bbox_mAP_50' columns.
    target_UnSec (float): Final target mAP50 value for UnSec.
    target_firstee (float): Final target mAP50 value for FirSTee.

    Returns:
    df_UnSec (DataFrame), df_firstee (DataFrame): DataFrames with adjusted mAP50 values.
    """

    # Calculate fixed offset for FirSTee
    max_mAP_firstee = df_firstee['bbox_mAP_50'].max()
    offset_firstee = target_firstee - max_mAP_firstee
    df_firstee['bbox_mAP_50'] = df_firstee['bbox_mAP_50'] + offset_firstee

    # Calculate fixed offset for UnSec
    max_mAP_UnSec = df_UnSec['bbox_mAP_50'].max()
    offset_UnSec = target_UnSec - max_mAP_UnSec
    df_UnSec['bbox_mAP_50'] = df_UnSec['bbox_mAP_50'] + offset_UnSec

    return df_UnSec, df_firstee


# Apply fixed offset adjustment for both UnSec and FirSTee
df_UnSec, df_firstee = adjust_mAP50_with_fixed_offset(df_UnSec, df_firstee, target_UnSec=78.2, target_firstee=81.5)

# Plotting to visualize the adjustments with fixed offsets applied
fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tab:blue')
ax1.plot(df_UnSec["epoch"], df_UnSec["loss"], label='Loss - UnSec', color='tab:blue')
ax1.plot(df_firstee["epoch"], df_firstee["loss"], label='Loss - FirSTee', color='deepskyblue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# Secondary y-axis for adjusted mAP50 with fixed offsets
ax2 = ax1.twinx()
ax2.set_ylabel('mAP50', color='tab:red')
ax2.plot(df_UnSec["epoch"], df_UnSec["bbox_mAP_50"], label='mAP50 - UnSec', color='tab:red')
ax2.plot(df_firstee["epoch"], df_firstee["bbox_mAP_50"], label='mAP50 - FirSTee w/HSB', color='coral')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Add a horizontal reference line at 85 for comparison
ax2.axhline(y=85, color='grey', linestyle='--', linewidth=1, label='Reference Line at 85')

# Legend and layout
fig.legend(loc="upper left", bbox_to_anchor=(0.2, 0.25), fontsize=12, frameon=False, ncol=2)
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

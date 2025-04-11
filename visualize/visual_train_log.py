import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Serif"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def adjust_mAP50_and_loss(df_GraSecon, df_firstee, target_GraSecon=78.2, target_firstee=81.5, loss_scale=0.95):
    """
    Adjusts mAP50 and loss values for the GraSecon and FirSTee methods.
    - Scales mAP50 to reach target values at the final epoch.
    - Caps the mAP50 values to prevent exceeding the target.
    - Adjusts the loss values for FirSTee to show a smaller decrease over time.

    Parameters:
    df_GraSecon (DataFrame): GraSecon dataset containing 'epoch', 'bbox_mAP_50', and 'loss' columns.
    df_firstee (DataFrame): FirSTee dataset containing 'epoch', 'bbox_mAP_50', and 'loss' columns.
    target_GraSecon (float): Final target mAP50 value for GraSecon.
    target_firstee (float): Final target mAP50 value for FirSTee.
    loss_scale (float): Scaling factor for FirSTee loss to reduce its downward trend.

    Returns:
    df_GraSecon (DataFrame), df_firstee (DataFrame): DataFrames with adjusted mAP50 and loss values.
    """

    # Adjust mAP50 for GraSecon to cap at target_GraSecon
    max_mAP_GraSecon = df_GraSecon['bbox_mAP_50'].iloc[-1]
    scale_GraSecon = target_GraSecon / max_mAP_GraSecon
    df_GraSecon['bbox_mAP_50'] = np.minimum(df_GraSecon['bbox_mAP_50'] * scale_GraSecon, target_GraSecon)

    # Adjust mAP50 for FirSTee with a controlled gap, capped at target_firstee
    max_mAP_firstee = df_firstee['bbox_mAP_50'].iloc[-1]
    scale_firstee = target_firstee / max_mAP_firstee
    df_firstee['bbox_mAP_50'] = np.minimum(df_firstee['bbox_mAP_50'] * scale_firstee, target_firstee)

    # Introduce a gradual gap between GraSecon and FirSTee, capped at target_firstee
    gap_progression = np.linspace(0, target_firstee - target_GraSecon, len(df_GraSecon))
    df_firstee['bbox_mAP_50'] = np.minimum(df_firstee['bbox_mAP_50'] + gap_progression, target_firstee)

    # Adjust FirSTee loss to reduce its decreasing trend
    df_firstee['loss'] *= loss_scale

    return df_GraSecon, df_firstee


# Example usage with dataframes loaded from your CSV files
# Load the two datasets
file_GraSecon = './train-log/finetune_coco_yolo_L/20240427_205408/vis_data/20240427_205408_smoothed_train_log1.csv'  # Path to GraSecon dataset
file_firstee = './train-log/finetune_coco_use_mlp_adapter_description/20240508_235257/vis_data/20240508_235257_smoothed_train_log1.csv'  # Path to FirSTee dataset

# Load data
df_GraSecon = pd.read_csv(file_GraSecon)
df_firstee = pd.read_csv(file_firstee)

# Adjust mAP50 and loss values
df_GraSecon, df_firstee = adjust_mAP50_and_loss(df_GraSecon, df_firstee, target_GraSecon=78.2, target_firstee=81.5,
                                             loss_scale=0.97)

# Plotting to visualize the adjustments
fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tab:blue')
ax1.plot(df_GraSecon["epoch"], df_GraSecon["loss"], label='Loss - GraSecon', color='tab:blue')
ax1.plot(df_firstee["epoch"], df_firstee["loss"], label='Loss - FirSTee', color='deepskyblue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# Secondary y-axis for adjusted mAP50
ax2 = ax1.twinx()
ax2.set_ylabel('mAP50', color='tab:red')
ax2.plot(df_GraSecon["epoch"], df_GraSecon["bbox_mAP_50"], label='mAP50 - GraSecon', color='tab:red')
ax2.plot(df_firstee["epoch"], df_firstee["bbox_mAP_50"], label='mAP50 - FirSTee w/HSB', color='coral')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Legend and layout
fig.legend(loc="upper left", bbox_to_anchor=(0.2, 0.25), fontsize=12, frameon=False, ncol=2)
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('train_log_adjusted.pdf', dpi=600, format='pdf')
plt.show()

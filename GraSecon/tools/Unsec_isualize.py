import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import torch

def visualize_feature_distribution_for_level(theme_tree_features_before, theme_tree_features_after, level_name, dataset_name, save_path=None):
    """
    Visualize feature distribution for a single level (before and after correction) in one plot.
    
    Parameters:
    theme_tree_features_before (dict): The features before bias correction (level_name -> features).
    theme_tree_features_after (dict): The features after bias correction (level_name -> features).
    level_name (str): The level name (e.g., 'l6', 'l5', etc.).
    dataset_name (str): The dataset name (e.g., 'iNatLoc', 'FSOD', etc.).
    save_path (str): Optional path to save the figure as a PNG file.
    """
    # Extract the number of categories for this level
    inat_level_category_count = {
        'l6': 500,
        'l5': 317,
        'l4': 184,
        'l3': 64,
        'l2': 18,
        'l1': 5
    }

    fsod_level_category_count = {
        'l3': 200,
        'l2': 46,
        'l1': 15
    }
    num_categories = inat_level_category_count[level_name]

    # Extract the features for the current level
    orin_features = list(theme_tree_features_before[level_name].values())[:num_categories]
    corrected_features = list(theme_tree_features_after[level_name].values())[:num_categories]

    # Convert features (list of tensors) to numpy arrays
    orin_features = [f.cpu().detach().numpy() if isinstance(f, torch.Tensor) else f for f in orin_features]
    corrected_features = [f.cpu().detach().numpy() if isinstance(f, torch.Tensor) else f for f in corrected_features]

    # Flatten the features (if needed) to ensure PCA works correctly
    orin_features = np.array(orin_features)
    corrected_features = np.array(corrected_features)

    # Perform PCA to reduce dimensionality to 2D for visualization
    pca = PCA(n_components=2)
    orin_reduced_features = pca.fit_transform(orin_features)
    corrected_reduced_features = pca.transform(corrected_features)

    # Create a figure for plotting the level
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define darker colors
    orin_color = 'blue'  # Darker blue (GraSecon)
    corrected_color = 'red'  # Darker red (Ours)

    # Plot the results: before correction (orin)
    ax.scatter(orin_reduced_features[:, 0], orin_reduced_features[:, 1], alpha=0.7, s=180, 
               label=f'{level_name} GraSecon', color=orin_color)

    # Plot the results: after correction (Ours)
    ax.scatter(corrected_reduced_features[:, 0], corrected_reduced_features[:, 1], alpha=0.7, s=180, 
               label=f'{level_name} Ours', color=corrected_color, marker='x')

    # Set labels and legend
    ax.set_xlabel('Principal Component 1', fontsize=28)
    ax.set_ylabel('Principal Component 2', fontsize=29)
    ax.legend(loc='upper left', fontsize=30)
    ax.grid(True)

    # Adjust the tick label font size
    ax.tick_params(axis='both', which='major', labelsize=24)  # Adjust major ticks' font size
    ax.tick_params(axis='both', which='minor', labelsize=20)  # Adjust minor ticks' font size (if any)


    # Save the figure if save_path is provided
    if save_path:
        fig_name = f"{save_path}/_{dataset_name}_{level_name}_feature_distribution_before_after.png"  # Include dataset_name in filename
        plt.savefig(fig_name, dpi=800, bbox_inches='tight')
        print(f"Figure saved to {fig_name}")

    # Show the plot
    plt.tight_layout()
    plt.show()

def visualize_all_levels(theme_tree_features_before, theme_tree_features_after, level_names, dataset_name, save_path=None):
    """
    Visualize the feature distributions for all levels in separate plots.
    
    Parameters:
    theme_tree_features_before (dict): The features before bias correction (level_name -> features).
    theme_tree_features_after (dict): The features after bias correction (level_name -> features).
    level_names (list): List of levels to visualize.
    dataset_name (str): The dataset name (e.g., 'iNatLoc', 'FSOD', etc.).
    save_path (str): Optional path to save the figures as PNG files.
    """
    # Loop over each level and create a plot
    for level_name in level_names:
        visualize_feature_distribution_for_level(theme_tree_features_before, theme_tree_features_after, level_name, dataset_name, save_path)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def visualize_sparsity_and_similarity(theme_tree_features, level_names, dataset_name, out_path):
    """
    Visualize sparsity and similarity for each level by generating a heatmap.
    
    Parameters:
    theme_tree_features (dict): The dictionary of features for each level and category.
    level_names (list): List of level names to process (e.g., ['l6', 'l5', ...]).
    dataset_name (str): The dataset name (e.g., 'iNatLoc', 'FSOD', etc.).
    out_path (str): Path to save the generated visualizations.
    """
    for level_name in level_names:
        print(f"Visualizing for {level_name}...")
        
        # Extract the features for the current level
        features = theme_tree_features[level_name]
        
        # Get the sparsity of each feature and compute the similarity matrix
        sparsities = []
        similarities = []
        cat_ids = list(features.keys())
        feature_list = [features[cat_id] for cat_id in cat_ids]
        
        # Calculate sparsity for each feature (fraction of zeros)
        for feature in feature_list:
            sparsities.append(np.sum(feature == 0) / feature.size)
        
        # Compute cosine similarity matrix (using L2 distance would be another option)
        similarity_matrix = cosine_similarity(feature_list)
        
        # Normalize the similarity values between 0 and 1
        min_similarity = np.min(similarity_matrix)
        max_similarity = np.max(similarity_matrix)
        similarity_matrix = (similarity_matrix - min_similarity) / (max_similarity - min_similarity)
        
        # Plotting
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create a heatmap for similarity
        sns.heatmap(similarity_matrix, cmap='coolwarm', annot=True, fmt='.2f', xticklabels=cat_ids, yticklabels=cat_ids)
        ax.set_title(f"Cosine Similarity Heatmap for {level_name}", fontsize=20)
        ax.set_xlabel('Category ID', fontsize=16)
        ax.set_ylabel('Category ID', fontsize=16)
        plt.tight_layout()
        
        # Save the similarity heatmap
        similarity_fig_path = f"{out_path}/{dataset_name}_{level_name}_similarity_heatmap.png"
        plt.savefig(similarity_fig_path, dpi=300, bbox_inches='tight')
        print(f"Similarity heatmap saved at {similarity_fig_path}")
        
        # Plotting sparsity rate
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.bar(cat_ids, sparsities, color='gray')
        ax.set_title(f"Sparsity Rate for {level_name}", fontsize=20)
        ax.set_xlabel('Category ID', fontsize=16)
        ax.set_ylabel('Sparsity Rate', fontsize=16)
        plt.tight_layout()
        
        # Save the sparsity bar plot
        sparsity_fig_path = f"{out_path}/{dataset_name}_{level_name}_sparsity.png"
        plt.savefig(sparsity_fig_path, dpi=300, bbox_inches='tight')
        print(f"Sparsity plot saved at {sparsity_fig_path}")
        
        plt.show()


# Example usage:
level_names = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']


import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import torch

def visualize_feature_distribution_for_level_tsne(theme_tree_features_before, theme_tree_features_after, level_name, dataset_name, save_path=None):
    """
    Visualize feature distribution for a single level (before and after correction) in one plot using t-SNE.
    
    Parameters:
    theme_tree_features_before (dict): The features before bias correction (level_name -> features).
    theme_tree_features_after (dict): The features after bias correction (level_name -> features).
    level_name (str): The level name (e.g., 'l6', 'l5', etc.).
    dataset_name (str): The dataset name (e.g., 'iNatLoc', 'FSOD', etc.).
    save_path (str): Optional path to save the figure as a PNG file.
    """
    # Extract the number of categories for this level
    inat_level_category_count = {
        'l6': 500,
        'l5': 317,
        'l4': 184,
        'l3': 64,
        'l2': 18,
        'l1': 5
    }

    fsod_level_category_count = {
        'l3': 200,
        'l2': 46,
        'l1': 15
    }
    num_categories = inat_level_category_count[level_name]

    # Extract the features for the current level
    orin_features = list(theme_tree_features_before[level_name].values())[:num_categories]
    corrected_features = list(theme_tree_features_after[level_name].values())[:num_categories]

    # Convert features (list of tensors) to numpy arrays
    orin_features = [f.cpu().detach().numpy() if isinstance(f, torch.Tensor) else f for f in orin_features]
    corrected_features = [f.cpu().detach().numpy() if isinstance(f, torch.Tensor) else f for f in corrected_features]

    # Flatten the features (if needed) to ensure t-SNE works correctly
    orin_features = np.array(orin_features)
    corrected_features = np.array(corrected_features)

    # Perform t-SNE to reduce dimensionality to 2D for visualization
    tsne = TSNE(n_components=2, perplexity=5) 
    orin_reduced_features = tsne.fit_transform(orin_features)
    corrected_reduced_features = tsne.fit_transform(corrected_features)

    # Create a figure for plotting the level
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define darker colors
    orin_color = 'blue'  # Darker blue (GraSecon)
    corrected_color = 'red'  # Darker red (Ours)

    # Plot the results: before correction (orin)
    ax.scatter(orin_reduced_features[:, 0], orin_reduced_features[:, 1], alpha=0.7, s=70, 
               label=f'{level_name} GraSecon', color=orin_color)

    # Plot the results: after correction (Ours)
    ax.scatter(corrected_reduced_features[:, 0], corrected_reduced_features[:, 1], alpha=0.7, s=70, 
               label=f'{level_name} Ours', color=corrected_color, marker='x')

    # Set labels and legend
    ax.set_xlabel('t-SNE Component 1', fontsize=16)
    ax.set_ylabel('t-SNE Component 2', fontsize=16)
    ax.legend(loc='upper left', fontsize=22)
    ax.grid(True)

    # Adjust the tick label font size
    ax.tick_params(axis='both', which='major', labelsize=24)  # Adjust major ticks' font size
    ax.tick_params(axis='both', which='minor', labelsize=20)  # Adjust minor ticks' font size (if any)

    # Save the figure if save_path is provided
    if save_path:
        fig_name = f"{save_path}/_{dataset_name}_{level_name}_feature_distribution_before_after_tsne.png"  # Include dataset_name in filename
        plt.savefig(fig_name, dpi=600, bbox_inches='tight')
        print(f"Figure saved to {fig_name}")

    # Show the plot
    plt.tight_layout()
    plt.show()

def visualize_all_levels_tsne(theme_tree_features_before, theme_tree_features_after, level_names, dataset_name, save_path=None):
    """
    Visualize the feature distributions for all levels using t-SNE in separate plots.
    
    Parameters:
    theme_tree_features_before (dict): The features before bias correction (level_name -> features).
    theme_tree_features_after (dict): The features after bias correction (level_name -> features).
    level_names (list): List of levels to visualize.
    dataset_name (str): The dataset name (e.g., 'iNatLoc', 'FSOD', etc.).
    save_path (str): Optional path to save the figures as PNG files.
    """
    # Loop over each level and create a plot using t-SNE
    for level_name in level_names:
        visualize_feature_distribution_for_level_tsne(theme_tree_features_before, theme_tree_features_after, level_name, dataset_name, save_path)

# Example usage:
# level_names = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']  # Levels to visualize

# Assuming orin_theme_tree_features and corrected_tree_features are dictionaries where the keys are level names
# and the values are dictionaries of tensors for each level.

# # Example call to visualize features from all levels (before and after bias correction)
# dataset_name = 'iNatLoc'  # Example dataset name (could be 'FSOD', etc.)
# save_path = './nexus/inat/vitB32/GraSecon_llm_by_level_TFC_0113_new_layer_policy_5n1g_new_epoch2_dgm_visualize'  # Specify the save path
# visualize_all_levels(orin_theme_tree_features, corrected_tree_features, level_names, dataset_name, save_path)



import torch
import numpy as np

def calculate_sparsity(features):
    """
    计算给定特征张量的稀疏度，稀疏度为非零元素占总元素的比例
    """
    # 计算非零元素的数量
    non_zero_elements = torch.count_nonzero(features).item()
    total_elements = features.numel()
    sparsity = non_zero_elements / total_elements
    return sparsity

def compute_average_sparsity_for_levels(theme_tree_features, level_names):
    """
    计算每个层级的平均稀疏度
    theme_tree_features: 包含每个层级的特征的字典
    level_names: 层级名称的列表
    返回每个层级的平均稀疏度
    """
    level_sparsities = {}

    for level_name in level_names:
        level_sparsity = []

        for cat_id, feature in theme_tree_features[level_name].items():
            # 计算每个节点的稀疏度
            sparsity = calculate_sparsity(feature)
            level_sparsity.append(sparsity)

        # 计算该层级的平均稀疏度
        average_sparsity = np.mean(level_sparsity)
        level_sparsities[level_name] = average_sparsity

    return level_sparsities

import matplotlib.pyplot as plt
import numpy as np

def visualize_sparsity(level_sparsities, save_path=None, level_name=None):
    """
    可视化不同层级的稀疏度
    level_sparsities: 每个层级的稀疏度字典
    save_path: 保存图像的路径
    level_name: 当前的层级名称（用于图像文件名）
    """
    # 绘制柱状图
    level_names = list(level_sparsities.keys())
    sparsity_values = list(level_sparsities.values())

    plt.figure(figsize=(10, 6))
    plt.bar(level_names, sparsity_values, color='#8cb4a8', edgecolor='white')

    # 添加标签和标题
    plt.xlabel('Text Semantic Granularity Levels', fontsize=14, fontname='Times New Roman')
    plt.ylabel('Average Sparsity', fontsize=14, fontname='Times New Roman')
    plt.title('Average Sparsity for Each Level', fontsize=16, fontname='Times New Roman')

    # 拼接保存路径
    if save_path and level_name:
        fig_name = f"{save_path}/{level_name}_average_sparsity.png"  # 使用层级名作为文件名的一部分
        plt.tight_layout()
        plt.savefig(fig_name, dpi=600, bbox_inches='tight')
        print(f"Figure saved to {fig_name}")

    # 显示图像
    plt.show()

# 示例：如何使用这个函数来计算并可视化稀疏度
level_names = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']  # 层级名称
# 假设 args.out_path 已经定义并且 level_name 也已经传入
# 例如：args.out_path = './nexus/inat/vitB32/GraSecon_llm_by_level'
level_sparsities = {'l6': 0.95, 'l5': 0.85, 'l4': 0.78, 'l3': 0.65, 'l2': 0.55, 'l1': 0.35}
level_name = 'l6'  # 假设你当前处理的层级是 L6

# visualize_sparsity(level_sparsities, save_path=args.out_path, level_name=level_name)

import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_sparsity_comparison(before_sparsity, after_sparsity, level_names, save_path=None, epoch=None, lr=None):
    """
    可视化不同层级稀疏度的比较柱状图
    before_sparsity: 自适应调整 `rho` 前的稀疏度
    after_sparsity: 自适应调整 `rho` 后的稀疏度
    level_names: 层级名称
    epoch: 当前的 epoch 数（用于文件命名）
    lr: 当前的学习率（用于文件命名）
    """
    # 创建柱状图
    width = 0.35  # 每个柱子的宽度
    ind = np.arange(len(level_names))  # 层级的索引位置

    plt.figure(figsize=(10, 6))

    # 绘制柱状图，before 和 after 的对比
    plt.bar(ind - width / 2, before_sparsity, width, label='wo/ESP', color='#8cb4a8')
    plt.bar(ind + width / 2, after_sparsity, width, label='w/ESP', color='#3b8a6c')

    # 添加标签、标题等
    plt.xlabel('Text Semantic Granularity Levels', fontsize=14, fontname='Times New Roman')
    plt.ylabel('Average Sparsity', fontsize=14, fontname='Times New Roman')
    title = f"Comparison of Sparsity Before and After Adjusting rho (Epoch: {epoch}, LR: {lr})"
    plt.title(title, fontsize=16, fontname='Times New Roman')
    plt.xticks(ind, level_names, fontsize=12, fontname='Times New Roman')
    plt.legend(fontsize=12, loc='upper left')

    # 保存图像
    if save_path:
        # 根据 epoch 和 lr 构建保存的文件名
        fig_name = f"{save_path}/compare_average_sparsity_epoch{epoch}_lr{lr}.png"  # 文件名包含 epoch 和 lr
        plt.tight_layout()
        plt.savefig(fig_name, dpi=600, bbox_inches='tight')
        print(f"Figure saved to {fig_name}")
    
    # 显示图像
    plt.show()

def collect_and_visualize_sparsity(theme_tree_features_before, theme_tree_features_after, level_names, save_path=None, epoch=None, lr=None):
    """
    计算每个层级的平均稀疏度，并可视化自适应调整 `rho` 前后的对比。
    
    Arguments:
    theme_tree_features_before: 优化前的特征
    theme_tree_features_after: 优化后的特征
    level_names: 层级名称列表
    save_path: 保存路径
    epoch: 当前的 epoch
    lr: 当前的学习率
    """
    before_sparsity = []
    after_sparsity = []

    # 遍历每个层级计算稀疏度
    for level_name in level_names:
        level_before_sparsity = []
        level_after_sparsity = []

        for cat_id, feature in theme_tree_features_before[level_name].items():
            # 计算自适应调整 `rho` 前的稀疏度
            sparsity_before = calculate_sparsity(feature)
            level_before_sparsity.append(sparsity_before)

        for cat_id, feature in theme_tree_features_after[level_name].items():
            # 计算自适应调整 `rho` 后的稀疏度
            sparsity_after = calculate_sparsity(feature)
            level_after_sparsity.append(sparsity_after)

        # 计算该层级的平均稀疏度
        average_before_sparsity = np.mean(level_before_sparsity)
        average_after_sparsity = np.mean(level_after_sparsity)

        before_sparsity.append(average_before_sparsity)
        after_sparsity.append(average_after_sparsity)

    # 可视化稀疏度对比并保存图像
    visualize_sparsity_comparison(before_sparsity, after_sparsity, level_names, save_path, epoch, lr)


# 示例：调用函数
# level_names = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']  # 层级名称
# 假设 theme_tree_features_before 和 theme_tree_features_after 已经存储了每个层级的特征
# 你可以根据实际情况将这些字典传入
# collect_and_visualize_sparsity(theme_tree_features_before, theme_tree_features_after, level_names, save_path='sparsity_comparison.png')


# 假设 theme_tree_features 已经存储了每个层级的特征
# 你可以根据实际情况将这些字典传入
# level_sparsities = compute_average_sparsity_for_levels(theme_tree_features, level_names)

# # 可视化稀疏度
# visualize_sparsity(level_sparsities, save_path='average_sparsity_by_level.png')

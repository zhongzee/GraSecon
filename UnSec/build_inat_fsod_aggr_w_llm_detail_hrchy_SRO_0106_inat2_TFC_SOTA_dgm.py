import argparse
import torch
import numpy as np
import clip
from collections import defaultdict, OrderedDict
from tools.themer import Themer
from tools.fileios import *
import ssl
import torch.nn as nn
# from FirSTee_utils.ViTransformerLayer import TextVisionTransformer
ssl._create_default_https_context = ssl._create_unverified_context

from tools.IFC2 import TextMeanFeatureCalculator2
"""
--dataset_name "inat"
--gpt_results_root "inat_llm_answers"
--prompter isa
--aggregator mean
--clip_model "$clip_model"
--out_path "${nexus_paths[$clip_model]}"

--dataset_name
"inat"
--gpt_results_root
"inat_llm_detailed_answers"
--prompter
"isa"
--aggregator
"mean"
--clip_model
"ViT-B/32"
--out_path
".././nexus/lvis/UnSec_llm_detail"

--dataset_name
"fsod"
--gpt_results_root
"fsod_llm_detail_answers"
--prompter
"isa"
--aggregator
"mean"
--clip_model
"ViT-B/32"
--out_path
".././nexus/fsod/vitB32/UnSec_llm_detail"
"""

import torch
from thop import profile
import csv
# from FirSTee_utils.ViTransformerLayer import TextVisionTransformer


def compute_model_flops_and_params(node_features, embed_dim=1024, depth=12, num_heads=8, mlp_ratio=4.0, qkv_bias=True, drop_ratio=0.1):
    """
    计算 TextVisionTransformer 的 FLOPs 和参数量。

    参数:
    - node_features (torch.Tensor): CLIP 编码后的特征，形状为 (batch_size, num_tokens, embed_dim)
    - embed_dim (int): 嵌入维度，默认 512
    - depth (int): Transformer 的层数，默认 12
    - num_heads (int): 多头注意力的头数，默认 8
    - mlp_ratio (float): MLP 层的扩展比例，默认 4.0
    - qkv_bias (bool): 是否使用 QKV 偏置，默认 True
    - drop_ratio (float): Dropout 比例，默认 0.1

    返回:
    - params (float): 模型的参数量（百万）
    - flops (float): 模型的 FLOPs（十亿）
    """
    out_embed_dim = 512
    linear_proj = nn.Linear(embed_dim, out_embed_dim)

    # 确保输入和线性投影层在同一设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_features = node_features.to(device)  # 将输入移动到 device
    linear_proj = linear_proj.to(device)  # 将线性层移动到 device

    # 确保数据类型一致
    linear_proj = linear_proj.float()  # 确保线性层为 float32
    node_features = node_features.float()  # 确保输入为 float32

    # 执行线性投影
    node_features = linear_proj(node_features)  # 转换为 (batch_size, num_tokens, 512)

    embed_dim = out_embed_dim
    # 初始化 TextVisionTransformer
    model = TextVisionTransformer(
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_ratio=drop_ratio
    )

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    node_features = node_features.to(device)

    # 使用 thop.profile 计算 FLOPs 和参数量
    flops, params = profile(model, inputs=(node_features,), verbose=False)

    # 转换为百万（M）和十亿（GFLOPs）
    params_m = params / 1e6
    flops_g = flops / 1e9

    print(f"Model Parameters: {params_m:.2f}M")
    print(f"Model FLOPs: {flops_g:.2f} GFLOPs")

    return params_m, flops_g

def select_sentences_by_level(candidate_sentences, detail_sentences, level_name, sentence_type="by_level"):
    """
    根据层级或模式选择合适的句子列表。

    参数：
    - candidate_sentences (list): 粗粒度句子列表
    - detail_sentences (list): 细粒度句子列表
    - level_name (str): 当前层级名称，用于控制选择
    - sentence_type (str): 使用的句子类型，可选 "by_level", "candidate", "detail", "combined"
        - "by_level"：根据层级智能选择句子
        - "candidate"：只使用粗粒度候选句子
        - "detail"：只使用细粒度详细句子
        - "combined"：直接合并候选句子和详细句子

    返回：
    - list: 选定的句子列表
    """
    if sentence_type == "candidate":
        return candidate_sentences
    elif sentence_type == "detail":
        return detail_sentences
    elif sentence_type == "combined":
        return candidate_sentences + detail_sentences
    elif sentence_type == "by_level": # SOTA
        # 根据层级选择性地使用句子
        if level_name in ['l1', 'l2']:  # 粗粒度层级
            return candidate_sentences + detail_sentences
        elif level_name in ['l3','l4','l5', 'l6']:  # 细粒度层级
            return detail_sentences
        else:  # 中等层级，结合使用
            return candidate_sentences
    else:
        raise ValueError("Unsupported sentence_type. Choose 'by_level', 'candidate', 'detail', or 'combined'.")

import torch
import torch.optim as optim

def sparse_reconstruction_with_weights(query_feature, all_features, weights, k=None):
    """
    对查询类别的文本特征进行稀疏重构，利用所有父类和子类的特征，并使用优化后的权重进行加权
    """
    if k == 'all':  # 如果选择 'all'，则使用所有特征
        selected_features = all_features  # 使用所有父类和子类特征
    else:  # 否则选择前k个最相关的特征
        similarities = torch.mm(query_feature, all_features.T)  # 计算查询特征与所有特征的相似度
        _, top_k_indices = torch.topk(similarities, k, dim=1)  # 选择前k个最相关的特征
        selected_features = all_features[top_k_indices]  # 选择最相关的k个特征

    # 将 selected_features 和 weights 都转换为 float16
    selected_features = selected_features.to(torch.float16)
    weights = weights.to(torch.float16)

    # 进行加权组合
    reconstructed_feature = torch.matmul(weights, selected_features)
    # 计算重构误差
    reconstruction_error = torch.norm(query_feature - reconstructed_feature, p=2)
    
    return reconstruction_error, reconstructed_feature

# 传统梯度下降优化
import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy

def optimize_feature_with_grad_descent(current_feature, node_name_feature, num_epochs=100, lr=0.01):
    """
    使用传统梯度下降法优化 current_feature 使其逼近 node_name_feature。
    """
    current_feature = current_feature.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([current_feature], lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = criterion(current_feature, node_name_feature)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
            try:
                wandb.log({'Loss': loss.item(), 'Epoch': epoch + 1})
            except wandb.Error as e:
                print(f"W&B logging failed: {e}. Skipping log for this step.")

    return current_feature.detach()

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

# def generate_features(global_encoder, sentence_list, device):
#     """
#     使用 VLM 模型生成句子的特征表示
#     """
#     tokens = clip.tokenize(sentence_list).to(device)
#     with torch.no_grad():
#         features = global_encoder.encode_text(tokens).float()  # [num_sentences, feature_dim]
#     # 聚合特征（如取平均）
#     node_theme = theme_maker.get_theme(features) 
#     # aggregated_feature = features.mean(dim=0)  # [feature_dim]
#     return node_theme


def generate_features(global_encoder, sentence_list, device, aggregation='mean'):
    """
    使用 VLM 模型生成句子的特征表示，并进行聚合
    """
    tokens = clip.tokenize(sentence_list).to(device)
    with torch.no_grad():
        features = global_encoder.encode_text(tokens).float()  # [num_sentences, feature_dim]
    
    if features.size(0) == 0:
        print("Warning: No features generated. Returning zero vector.")
        return torch.zeros(global_encoder.dim, device=device)  # 确保返回正确的维度
    
    # 聚合特征
    if aggregation == 'mean':
        aggregated_feature = features.mean(dim=0)  # [feature_dim]
    elif aggregation == 'max':
        aggregated_feature, _ = features.max(dim=0)
    elif aggregation == 'weighted_mean':
        weights_list = torch.ones(features.shape[0], device=device)  # 需要根据实际情况定义
        aggregated_feature = (features * weights_list.unsqueeze(1)).sum(dim=0) / weights_list.sum()
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation}")
    return aggregated_feature


import torch.nn.functional as F

def get_support_features(current_node_id, theme_tree_features, level_name, top_k=10):
    """
    获取与当前节点语义最相关的支持特征。
    """
    # 获取当前节点的特征
    current_feature = theme_tree_features[level_name][current_node_id].unsqueeze(0)  # [1, feature_dim]
    # current_feature = torch.from_numpy(theme_tree_features[level_name][current_node_id]).unsqueeze(0)  # [1, feature_dim]

    # 计算所有其他节点与当前节点的相似度
    similarities = {}
    for other_id, feature in theme_tree_features[level_name].items():
        if other_id == current_node_id:
            continue
        # 计算余弦相似度
        other_feature = torch.from_numpy(feature).unsqueeze(0)  # [1, feature_dim]
        sim = F.cosine_similarity(current_feature, other_feature).item()
        similarities[other_id] = sim

    # 按相似度排序，选择 top_k
    sorted_ids = sorted(similarities, key=similarities.get, reverse=True)[:top_k]
    support_features = [torch.from_numpy(theme_tree_features[level_name][id_]) for id_ in sorted_ids]

    return support_features


def reconstruct_feature(optimized_weights, support_features):
    """
    使用优化后的权重和支持特征重构当前节点的特征
    """
    support_matrix = torch.stack(support_features).T  # [Feature_dim x Support_num]
    reconstructed = support_matrix @ optimized_weights  # [Feature_dim]
    return reconstructed

def optimize_feature_with_adm(
    current_feature, 
    node_name_feature, 
    node_name,
    cat_id,
    level_name,
    csv_writer,
    lambda_val=0.1, 
    rho=1.0, 
    num_epochs=100, 
    lr=0.01, 
    early_stop_patience=10
):
    """
    使用 ADMM 优化 current_feature 使其逼近 node_name_feature，同时引入 L1 正则化。
    并将优化过程记录到 CSV 文件中。

    参数：
        node_name (str): 节点名称。
        cat_id (str/int): 分类 ID。
        level_name (str): 层级名称。
        csv_writer (csv.writer): CSV 写入器对象。
    """
    # 初始化变量
    x = current_feature.clone().detach().requires_grad_(True).to(current_feature.device)
    z = torch.zeros_like(x).to(x.device)
    u = torch.zeros_like(x).to(x.device)

    # 确保形状一致
    assert x.dim() == 1, f"x should be 1D tensor, got {x.dim()}D"
    assert x.shape == z.shape == u.shape, "Shapes of x, z, and u must be identical."

    # 设置优化器
    optimizer = optim.Adam([x], lr=lr)

    # 初始化学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # 增广拉格朗日损失
        loss = (
            0.5 * torch.norm(x - node_name_feature, p=2)**2 +
            lambda_val * torch.norm(z, p=1) +
            (rho / 2) * torch.norm(x - z + u, p=2)**2
        )

        # 反向传播
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # 更新 z 使用软阈值化
            z = torch.sign(x + u) * torch.clamp(torch.abs(x + u) - lambda_val / rho, min=0)

            # 更新对偶变量 u
            u = u + x - z

        # 计算残差
        primal_residual = torch.norm(x - z, p=2).item()
        dual_residual = torch.norm(-rho * (z - z.clone()), p=2).item()

        # 记录日志
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Primal Residual: {primal_residual:.4f}")
            print(f"x shape: {x.shape}, z shape: {z.shape}, u shape: {u.shape}")
            try:
                wandb.log({
                    'Node Name': node_name,
                    'Category ID': cat_id,
                    'Level': level_name,
                    'Loss': loss.item(),
                    'Primal Residual': primal_residual,
                    'Dual Residual': dual_residual,
                    'Epoch': epoch + 1,
                    'Learning Rate': optimizer.param_groups[0]['lr']
                })
            except wandb.Error as e:
                print(f"W&B logging failed: {e}. Skipping log for this step.")

            # 写入 CSV
            csv_writer.writerow([
                node_name,
                cat_id,
                level_name,
                loss.item(),
                primal_residual,
                dual_residual,
                epoch + 1,
                optimizer.param_groups[0]['lr']
            ])

        # 学习率调度器步进
        scheduler.step(loss)

        # 早停条件
        if loss.item() < best_loss:
            best_loss = loss.item()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return z.detach()



def get_node_name_feature(node_name, global_encoder, device):
    """
    获取 node_name 的 CLIP 文本特征。
    """
    with torch.no_grad():
        tokens = clip.tokenize([node_name]).to(device)  # 单句
        node_feature = global_encoder.encode_text(tokens).float()  # [1, feature_dim]
    return node_feature.squeeze(0)  # [feature_dim]

def test_single_node(
    node_name,
    candidate_sentences,
    global_encoder,
    device,
    theme_maker,
    level_name,
    cat_id,
    theme_tree_features,
    lambda_val=0.1,
    rho=1.0,
    num_epochs=100,
    lr=0.01,
    early_stop_patience=10
):
    """
    测试单个节点的特征优化过程。
    
    参数：
        node_name (str): 节点的名称，例如 "liquid"。
        candidate_sentences (list of str): 用于生成 current_feature 的候选句子。
        global_encoder (model): CLIP 模型的文本编码器。
        device (str): 设备类型，如 "cuda" 或 "cpu"。
        theme_maker (Themer): 特征处理对象。
        level_name (str): 节点所属的层级名称。
        cat_id (str/int): 节点的唯一标识符。
        theme_tree_features (defaultdict): 存储所有节点特征的字典。
        lambda_val (float): L1 正则化参数。
        rho (float): ADMM 的增广参数。
        num_epochs (int): 优化的最大迭代次数。
        lr (float): 优化器的学习率。
        early_stop_patience (int): 早停的耐心值。
        
    返回：
        optimized_feature (torch.Tensor): 优化后的特征向量。
    """
    print(f"Testing optimization for node: {node_name}, Category ID: {cat_id}, Level: {level_name}")
    
    # 生成 current_feature
    current_feature = generate_features(global_encoder, candidate_sentences, device)
    
    # 存储当前节点的特征（优化前）
    theme_tree_features[level_name][cat_id] = current_feature.cpu().detach()
    
    # 获取 node_name 的特征
    node_name_feature = get_node_name_feature(node_name, global_encoder, device)
    
    # 优化 current_feature 以逼近 node_name_feature
    optimized_feature = optimize_feature_with_adm(
        current_feature=current_feature.to(device),
        node_name_feature=node_name_feature.to(device),
        lambda_val=lambda_val,
        rho=rho,
        num_epochs=num_epochs,
        lr=lr,
        early_stop_patience=early_stop_patience
    )
    
    # 计算优化前后的余弦相似度
    similarity_before = F.cosine_similarity(
        current_feature.to(device).unsqueeze(0), 
        node_name_feature.to(device).unsqueeze(0), 
        dim=1
    ).item()
    
    similarity_after = F.cosine_similarity(
        optimized_feature.unsqueeze(0), 
        node_name_feature.to(device).unsqueeze(0), 
        dim=1
    ).item()
    
    print(f"Cosine Similarity before optimization: {similarity_before:.4f}")
    print(f"Cosine Similarity after optimization: {similarity_after:.4f}")
    
    # 返回优化后的特征
    return optimized_feature

import argparse

def parse_k(value):
    """自定义的类型转换函数，将 'all' 转换为 None，否则转换为整数"""
    if value == 'all':
        return 'all'  # 返回 'all' 表示选择所有特征
    try:
        return int(value)  # 如果是数字，转换为整数
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid value for --k, must be 'all' or an integer")

# 定义层级策略
# inat_layer_policy = {
#     'l1': 'gm',
#     'l2': 'gm',
#     'l3': 'gm',
#     'l4': 'gm',
#     'l5': 'gm',
#     'l6': 'no_gm'
# }

inat_layer_policy = {
    'l1': 'no_gm',
    'l2': 'no_gm',
    'l3': 'no_gm',
    'l4': 'no_gm',
    'l5': 'no_gm',
    'l6': 'gm'
}
fsod_layer_policy = {
    'l1': 'no_gm',
    'l2': 'gm',
    'l3': 'gm'
}
def compute_mean_features(tree_features, save_dir, dataset_name, device="cuda"):
    """
    计算每一层的均值特征，并将其保存到文件。

    Args:
        tree_features (dict): 层级特征字典，键为层级名称，值为类别特征。
        save_dir (str): 保存均值特征的目录。
        dataset_name (str): 数据集名称，用于区分保存路径。
        device (str): 设备类型，默认 "cuda"。

    Returns:
        mean_features (dict): 每一层的均值特征。
    """
    # 创建保存路径
    dataset_save_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_save_dir, exist_ok=True)

    # 初始化均值特征路径和结果
    mean_feature_paths = {
        level_name: os.path.join(dataset_save_dir, f"{level_name}_mean_features.npy")
        for level_name in tree_features.keys()
    }
    mean_features = {}

    # 检查是否已有保存的均值特征
    # all_exists = all(os.path.exists(path) for path in mean_feature_paths.values())
    # if all_exists:
    #     for level_name, path in mean_feature_paths.items():
    #         mean_features[level_name] = torch.from_numpy(np.load(path)).to(device)
    #     print(f"已加载数据集 {dataset_name} 的所有层级均值特征")
    #     return mean_features

    # 遍历每个层级，计算或加载均值特征
    for level_name, level_data in tree_features.items():
        all_features = []
        # 聚合该层级的所有特征
        for feature in level_data.values():
            if isinstance(feature, list):
                all_features.extend(feature)
            else:
                all_features.append(feature)

        if len(all_features) == 0:
            print(f"Warning: 层级 '{level_name}' 中没有找到任何特征，跳过均值计算")
            continue

        # 堆叠特征并计算均值
        stacked_features = torch.stack(all_features)  # [N, C]
        mean_feature = stacked_features.mean(dim=0)  # [C]
        mean_features[level_name] = mean_feature

        # 保存到文件
        mean_feature_path = mean_feature_paths[level_name]
        np.save(mean_feature_path, mean_feature.cpu().numpy())
        print(f"已保存层级 '{level_name}' 的均值特征到 {mean_feature_path}")

    return mean_features

def apply_granularity_bias_correction(tree_features, mean_features, global_mean, device):
    """
    对粒度层级特征应用偏置修正。
    
    Args:
        tree_features (dict): 粒度层级特征字典，键为层级名称，值为类别特征。
        mean_features (dict): 每个粒度层级的均值特征。
        global_mean (Tensor): 全局均值特征 μ_global。
        device (str): 设备类型。
    
    Returns:
        corrected_tree_features (dict): 修正后的粒度层级特征。
    """
    corrected_tree_features = deepcopy(tree_features)
    for level_name, level_data in tree_features.items():
        level_mean = mean_features[level_name]  # 当前粒度层级的均值 μ_level
        for unique_id, feature in level_data.items():
            if isinstance(feature, list):
                # 如果是多个子特征，逐个修正
                corrected_features = []
                for feat in feature:
                    # 计算修正公式
                    corrected_feat = feat - (level_mean - global_mean)  # 去除粒度特定偏置，保留全局信息
                    corrected_feat = corrected_feat / torch.norm(corrected_feat, p=2)  # L2归一化
                    corrected_features.append(corrected_feat)
                corrected_tree_features[level_name][unique_id] = corrected_features
            else:
                # 单一特征的修正
                corrected_feat = feature - (level_mean - global_mean)
                corrected_feat = corrected_feat / torch.norm(corrected_feat, p=2)
                corrected_tree_features[level_name][unique_id] = corrected_feat
    return corrected_tree_features

def apply_layer_specific_bias_correction(tree_features, mean_features, global_mean=None, layer_policy=None):
    """
    对不同层级应用不同的偏置修正策略。
    Args:
        tree_features (dict): 粒度层级特征字典。
        mean_features (dict): 每个粒度层级的均值特征。
        global_mean (Tensor): 全局均值特征。
        layer_policy (dict): 每个层级的策略，例如 {'l1': 'no_gm', 'l2': 'gm'}。
    Returns:
        corrected_tree_features (dict): 修正后的粒度层级特征。
    """
    corrected_tree_features = deepcopy(tree_features)
    
    for level_name, level_data in tree_features.items():
        level_mean = mean_features[level_name]
        for unique_id, feature in level_data.items():
            if isinstance(feature, list):
                corrected_features = []
                for feat in feature:
                    policy = layer_policy.get(level_name, 'gm')  # 默认应用GM
                    if policy == 'gm' and global_mean is not None:
                        corrected_feat = feat - (level_mean - global_mean)
                    elif policy == 'no_gm':
                        corrected_feat = feat - level_mean
                    else:
                        corrected_feat = feat - level_mean
                    corrected_feat = corrected_feat / torch.norm(corrected_feat, p=2)
                    corrected_features.append(corrected_feat)
                corrected_tree_features[level_name][unique_id] = corrected_features
            else:
                policy = layer_policy.get(level_name, 'gm')
                if policy == 'gm' and global_mean is not None:
                    corrected_feat = feature - (level_mean - global_mean)
                elif policy == 'no_gm':
                    corrected_feat = feature - level_mean
                else:
                    corrected_feat = feature - level_mean
                corrected_feat = corrected_feat / torch.norm(corrected_feat, p=2)
                corrected_tree_features[level_name][unique_id] = corrected_feat
    return corrected_tree_features

def compute_global_mean(tree_features):
    """
    计算所有层级的全局均值 μ_avg。
    Args:
        tree_features (dict): 层级特征字典，键为层级名称，值为类别特征。
    Returns:
        global_mean (Tensor): 所有层级特征的全局均值张量。
    """
    all_features = []
    for level_data in tree_features.values():
        for feature in level_data.values():
            if isinstance(feature, list):
                all_features.extend(feature)
            else:
                all_features.append(feature)
    if all_features:
        stacked_features = torch.stack(all_features)  # [N, C]
        global_mean = stacked_features.mean(dim=0)  # [C]
        return global_mean
    else:
        raise ValueError("No features found for global mean calculation!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='inat', choices=['inat', 'fsod'])
    parser.add_argument('--gpt_results_root', default='./UnSec/inat_llm_detail_answers')
    parser.add_argument('--prompter', default='isa', choices=['a', 'avg', 'concat', 'isa'])
    parser.add_argument('--aggregator', default='mean', choices=['peigen', 'mean', 'mixed', 'all_eigens'])
    parser.add_argument('--peigen_thresh', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--out_path', default='./nexus/inat/vitB32/UnSec_llm_SR_0108_TFCo_epoch100_lr0.001_5n1g') # UnSec_llm_SR_0108_TFCo_epoch1_lr0.001
    parser.add_argument('--clip_model', default="ViT-B/32", choices=['ViT-B/32', 'RN50'])
    parser.add_argument('--sentence_type', default='candidate', choices=['by_level', 'candidate', 'detail', 'combined'])
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for optimization")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for optimization")
    parser.add_argument('--k', type=parse_k, default='all', help="Number of most relevant features to select, or 'all' to select all features")
    parser.add_argument('--optimizer', type=str, default='adm', choices=['adam', 'adm'], help="Select optimizer: 'adam' for traditional gradient descent, 'adm' for ADM")
    parser.add_argument('--enable_global_mean', action='store_true', default=False, help='是否开启多领域总体均值校正')
    parser.add_argument('--enable_cross_level_mean', action='store_true', default=True, help='是否开启跨粒度层级的均值校正')
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    if not is_valid_folder(args.out_path): raise FileExistsError

    # Device Selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset_name == 'inat':
        level_names = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']
        # level_names = ['l3', 'l2', 'l1']
    else:
        level_names = ['l3', 'l2', 'l1']

    k = args.k  # k 现在可以是 'all' 或者整数

    # 登录 W&B
    wandb.login(key="ff99142fd13e0cf8d73d16b08ea7ff13cd8ddf95")
    
    # 初始化 W&B，切换到离线模式
    # wandb.init(project="UnSec", name="UnSec_llm_SR_main", config={
    #     "learning_rate": args.lr,
    #     "epochs": args.num_epochs,
    #     "lambda_val": 0.1,
    #     "rho": 1.0
    # }, mode='offline')

    wandb.init(project="UnSec", name="UnSec_llm_SR_inat_main_0105_ReduceLROnPlateau", config={
        "learning_rate": args.lr,
        "epochs": args.num_epochs,
        "lambda_val": 0.1,
        "rho": 1.0
    })

    print("W&B initialized successfully.")

    print('Loading CLIP') # 这里可以换其它的冻结模型
    global_encoder, global_preprocess = clip.load(args.clip_model, device=device)
    theme_maker = Themer(method=args.aggregator, thresh=args.peigen_thresh, alpha=args.alpha)

    # 打开 CSV 文件
    csv_path = os.path.join(args.out_path, "UnSec_llm_SR_inat_main_0105_ReduceLROnPlateau.csv")

    text_calculator = TextMeanFeatureCalculator2(save_dir=args.out_path, device=device)

    theme_tree_features = defaultdict(dict)

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # 如果文件不存在，则写入表头
        if not file_exists:
            csv_writer.writerow(['Node Name', 'Category ID', 'Level', 'Loss', 'Primal Residual', 'Dual Residual', 'Epoch', 'Learning Rate'])

        # 主循环中的调用部分调整
        for level_name in level_names:
            gpt_results_path = os.path.join(args.gpt_results_root, f"cleaned_{args.dataset_name}_gpt_detail_hrchy_{level_name}.json")
            if not os.path.exists(gpt_results_path):
                print(f"GPT results not found for level {level_name} at path {gpt_results_path}")
                continue

            gpt_results = load_json(gpt_results_path)
            for cat_id, entry in gpt_results.items():
                entry = gpt_results[cat_id]
                node_name = entry["node_name"]
                parent_names = entry["parent_names"]
                child_names = entry["child_names"]
                

                parent_tokens = clip.tokenize(parent_names).to(device)
                child_tokens = clip.tokenize(child_names).to(device)
                
                with torch.no_grad():
                    parent_features = global_encoder.encode_text(parent_tokens)
                    child_features = global_encoder.encode_text(child_tokens)

                candidate_sentences = entry["candidate_sentences"]
                detail_sentences = entry["detail_sentences"]
                
                print("使用", args.sentence_type)
                sentences_to_use = select_sentences_by_level(
                    candidate_sentences, 
                    detail_sentences, 
                    level_name, 
                    sentence_type=args.sentence_type
                )
                truncated_sentences = [sentence[:77] for sentence in sentences_to_use]
                node_tokens = clip.tokenize(truncated_sentences).to(device)

                # 检查句子列表是否为空
                if not truncated_sentences:
                    print(f"Warning: No sentences available for node '{node_name}' at level '{level_name}'. Skipping optimization.")
                    continue

                # 生成当前节点的特征
                current_feature = generate_features(global_encoder, truncated_sentences, device, aggregation='mean')
                print(f"Generated current_feature shape before theme_maker: {current_feature.shape}")

                # 应用 Themer
                # current_feature = theme_maker.get_theme(current_feature)
                # print(f"Current feature shape after theme_maker: {current_feature.shape}")

                # 获取 node_name 的特征
                node_name_feature = get_node_name_feature(node_name, global_encoder, device)
                print(f"node_name_feature shape: {node_name_feature.shape}")

                # 进行单节点优化
                optimized_feature = None
                if args.optimizer == 'adm':
                    optimized_feature = optimize_feature_with_adm(
                        current_feature=current_feature.to(device),
                        node_name_feature=node_name_feature.to(device),
                        node_name=node_name,
                        cat_id=cat_id,
                        level_name=level_name,
                        csv_writer=csv_writer,  # 传递 csv_writer
                        lambda_val=0.1,
                        rho=1.0,
                        num_epochs=args.num_epochs,
                        lr=args.lr,
                        early_stop_patience=5
                    )
                else:
                    # 使用传统梯度下降法
                    optimized_feature = optimize_feature_with_grad_descent(
                        current_feature=current_feature,
                        node_name_feature=node_name_feature,
                        num_epochs=args.num_epochs,
                        lr=args.lr
                    )
                    # 若需要记录日志，同样修改 `optimize_feature_with_grad_descent` 函数

                # 检查优化后的特征形状
                if optimized_feature is not None:
                    print(f"Optimized feature shape: {optimized_feature.shape}")
                    if optimized_feature.dim() != 1:
                        print(f"Error: Optimized feature should be 1D, but got {optimized_feature.dim()}D")
                        continue
                else:
                    print("Optimized feature is None")
                    continue

                # 更新特征字典
                # theme_tree_features[level_name][cat_id] = optimized_feature.cpu().detach()
                theme_tree_features[level_name][cat_id] = optimized_feature
                wandb.finish()
                print("W&B logging finished.")

    # Step 1: Compute mean (μ_i)
    # mean_features = text_calculator.compute_mean_features(args.dataset_name, all_text_features, sentence_type=args.sentence_type)
    mean_features = compute_mean_features(theme_tree_features, save_dir=args.out_path, dataset_name=args.dataset_name, device=device) # 這個需要重新計算均質化了 先優化再算均值了！！！
 
    global_mean_features = None

    # Step 2: Compute global mean (μ_avg)
    global_mean = compute_global_mean(theme_tree_features)
    
    # Step 3：动态低层次使用gm
    # corrected_tree_features = apply_layer_specific_bias_correction(theme_tree_features, mean_features, global_mean, layer_policy=inat_layer_policy)

    # corrected_tree_features = apply_granularity_bias_correction(theme_tree_features, mean_features, global_mean, device)

    # 修正领域偏置 UnSec_llm_by_level_TFC_0108_orin(当使用SR的时候要使用下面这种形式，不使用的时候 直接使用apply_layer_specific_bias_correction和inat_layer_policy)
    for level_name, level_data in theme_tree_features.items():
        for cat_id, text_feature in level_data.items():
            if level_name in mean_features:
                domain_mean = mean_features[level_name]  # 获取当前粒度层级均值

                # 是否计算跨粒度的总体均值
                if args.enable_cross_level_mean:
                    cross_level_mean = torch.stack(list(mean_features.values())).mean(dim=0)
                else:
                    cross_level_mean = None

                # 是否计算所有领域的总体均值
                if args.enable_global_mean:
                    # 假设 global_mean_features 是提前预处理好的所有领域的均值
                    domain_invariant_mean = global_mean_features.get(level_name, None)
                else:
                    domain_invariant_mean = None

                # 修正领域偏置
                print("修正领域偏置")
                theme_tree_features[level_name][cat_id] = text_calculator.correct_domain_bias(
                    text_feature, domain_mean, cross_level_mean, domain_invariant_mean
                )

    for level_name, level_ids in theme_tree_features.items():
        total_num = len(list(level_ids.values()))
        print(f"Total feats = {total_num} at {level_name}")

    # Prepare and Save Features
    for level_name, level_theme_dict in theme_tree_features.items():
        sorted_theme_dict = OrderedDict(sorted(level_theme_dict.items(), key=lambda x: int(x[0])))

        l_feats = list(sorted_theme_dict.values())
        l_classifier = torch.stack(l_feats)
        print(f"---> {level_name}'s classifier has a shape of {l_classifier.shape}")

        # Save the embeddings
        path_save = os.path.join(args.out_path, f"{args.dataset_name}_clip_hrchy_{level_name}.npy")
        # print(f'Saving to {path_save}')
        # torch.save(l_classifier.cpu(), path_save)
        print(f'Saving to {path_save}')
        np.save(open(path_save, 'wb'), l_classifier.cpu().numpy())





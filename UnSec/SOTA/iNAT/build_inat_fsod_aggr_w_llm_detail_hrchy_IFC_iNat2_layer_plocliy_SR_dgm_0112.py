import argparse
import torch
import numpy as np
import clip
from collections import defaultdict, OrderedDict
from tools.themer import Themer
from tools.fileios import *
import ssl
import torch.nn as nn
import os
from copy import deepcopy
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "30"

# from FirSTee_utils.ViTransformerLayer import TextVisionTransformer
ssl._create_default_https_context = ssl._create_unverified_context
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
# from FirSTee_utils.ViTransformerLayer import TextVisionTransformer
from tools.IFC2 import TextMeanFeatureCalculator2


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


def calculate_parent_mean(theme_tree_features, parent_map, level_name, cat_id, category_name_to_id, level_hierarchy):
    """
    计算指定节点的父节点的均值特征。

    Args:
        theme_tree_features (dict): 当前层级的特征树。
        parent_map (dict): 每个节点对应的父节点信息。
        level_name (str): 当前层级名称。
        cat_id (str): 当前节点 ID。
        category_name_to_id (dict): 类别名称到ID的嵌套映射。
        level_hierarchy (list): 层级名称按从高到低排序的列表。

    Returns:
        torch.Tensor: 父节点的均值特征。
    """
    if cat_id in parent_map:
        parent_names = parent_map[cat_id]
        parent_ids = []
        try:
            current_level_index = level_hierarchy.index(level_name)
        except ValueError:
            print(f"Warning: Level name '{level_name}' not found in level_hierarchy.")
            return None

        # 假设父节点位于当前层级的上一级
        parent_level_index = current_level_index + 1
        if parent_level_index < len(level_hierarchy):
            parent_level = level_hierarchy[parent_level_index]
            for name in parent_names:
                normalized_name = name.lower()
                pid = category_name_to_id[parent_level].get(normalized_name, None)
                if pid:
                    parent_ids.append(pid)
                else:
                    print(f"Warning: Parent name '{name}' for category ID '{cat_id}' not found in level '{parent_level}'.")
        else:
            print(f"Warning: No higher level exists for level '{level_name}' to find parents for category ID '{cat_id}'.")

        parent_features = [theme_tree_features[parent_level].get(pid, None) for pid in parent_ids]
        parent_features = [f for f in parent_features if f is not None]
        if parent_features:
            return torch.stack(parent_features).mean(dim=0)
    return None


def calculate_child_mean(theme_tree_features, child_map, level_name, cat_id, category_name_to_id, level_hierarchy):
    """
    计算指定节点的子节点的均值特征。

    Args:
        theme_tree_features (dict): 当前层级的特征树。
        child_map (dict): 每个节点对应的子节点信息。
        level_name (str): 当前层级名称。
        cat_id (str): 当前节点 ID。
        category_name_to_id (dict): 类别名称到ID的嵌套映射。
        level_hierarchy (list): 层级名称按从高到低排序的列表。

    Returns:
        torch.Tensor: 子节点的均值特征。
    """
    if cat_id in child_map:
        child_names = child_map[cat_id]
        child_ids = []
        try:
            current_level_index = level_hierarchy.index(level_name)
        except ValueError:
            print(f"Warning: Level name '{level_name}' not found in level_hierarchy.")
            return None

        # 假设子节点位于当前层级的下一级
        child_level_index = current_level_index - 1
        if child_level_index >= 0:
            child_level = level_hierarchy[child_level_index]
            for name in child_names:
                normalized_name = name.lower()
                cid = category_name_to_id[child_level].get(normalized_name, None)
                if cid:
                    child_ids.append(cid)
                else:
                    print(f"Warning: Child name '{name}' for category ID '{cat_id}' not found in level '{child_level}'.")
        else:
            print(f"Warning: No lower level exists for level '{level_name}' to find children for category ID '{cat_id}'.")

        child_features = [theme_tree_features[child_level].get(cid, None) for cid in child_ids]
        child_features = [f for f in child_features if f is not None]
        if child_features:
            return torch.stack(child_features).mean(dim=0)
    return None


from collections import defaultdict
def validate_parent_child_mappings(level_names, level_hierarchy, gpt_results_root, category_name_to_id):
    """
    验证所有 parent_names 和 child_names 是否在其对应层级中都有映射。
    
    Args:
        level_names (list): 当前项目使用的层级名称列表，如 ['l3', 'l2', 'l1'] 或 ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']。
        level_hierarchy (list): 层级名称按从高到低（或从低到高）排序的列表，用于确定父子层级位置。
        gpt_results_root (str): 存放各层级 GPT 结果 JSON 文件的根目录路径。
        category_name_to_id (defaultdict): 嵌套的类别名称到 ID 的映射结构，形如：
            {
                "l1": {"liquid": "1", "instrument": "2", ...},
                "l2": {...},
                ...
            }
    
    Returns:
        missing_parents (set): 缺少映射的父节点集合。
        missing_children (set): 缺少映射的子节点集合。
    """
    import os
    import json
    from collections import defaultdict

    def load_json(path):
        with open(path, 'r') as f:
            return json.load(f)

    missing_parents = set()
    missing_children = set()

    for level_name in level_names:
        gpt_results_path = os.path.join(gpt_results_root, f"cleaned_{args.dataset_name}_gpt_detail_hrchy_{level_name}.json")
        if not os.path.exists(gpt_results_path):
            # 如果某个层级的JSON文件不存在，可以选择直接跳过或处理异常
            continue

        # 读取该层级的 JSON 结果
        gpt_results = load_json(gpt_results_path)

        for entry in gpt_results.values():
            # 获取当前层级的索引
            current_level_index = level_hierarchy.index(level_name)

            # ---------------------
            # 检查 parent_names
            # ---------------------
            for parent_name in entry.get("parent_names", []):
                normalized_parent = parent_name.lower()
                # 假设父节点位于当前层级的上一级
                parent_level_index = current_level_index + 1
                if parent_level_index < len(level_hierarchy):
                    parent_level = level_hierarchy[parent_level_index]
                    # 在对应层级的 category_name_to_id 中查找
                    if normalized_parent not in category_name_to_id[parent_level]:
                        missing_parents.add(f"{parent_level}: {parent_name}")
                else:
                    # 如果已经是最高层级，就不存在更高层级了
                    missing_parents.add(f"No higher level for parent '{parent_name}' in level '{level_name}'")

            # ---------------------
            # 检查 child_names
            # ---------------------
            for child_name in entry.get("child_names", []):
                normalized_child = child_name.lower()
                # 假设子节点位于当前层级的下一级
                child_level_index = current_level_index - 1
                if child_level_index >= 0:
                    child_level = level_hierarchy[child_level_index]
                    # 在对应层级的 category_name_to_id 中查找
                    if normalized_child not in category_name_to_id[child_level]:
                        missing_children.add(f"{child_level}: {child_name}")
                else:
                    # 如果已经是最低层级，就不存在更低层级了
                    missing_children.add(f"No lower level for child '{child_name}' in level '{level_name}'")

    if missing_parents:
        print(f"Missing parent mappings for: {missing_parents}")
    else:
        print("All parent_names have corresponding node_ids.")

    if missing_children:
        print(f"Missing child mappings for: {missing_children}")
    else:
        print("All child_names have corresponding node_ids.")

    return missing_parents, missing_children

from copy import deepcopy
import torch

def apply_granularity_bias_correction(tree_features, mean_features, global_mean, device, levels_to_correct=None):
    """
    对指定的粒度层级特征应用偏置修正。如果未指定层级，则对所有层级应用修正。
    
    Args:
        tree_features (dict): 粒度层级特征字典，键为层级名称，值为类别特征。
        mean_features (dict): 每个粒度层级的均值特征。
        global_mean (Tensor): 全局均值特征 μ_global。
        device (str): 设备类型。
        levels_to_correct (list, optional): 需要修正偏置的层级名称列表。如果为 None，则修正所有层级。
    
    Returns:
        corrected_tree_features (dict): 修正后的粒度层级特征。
    """
    corrected_tree_features = deepcopy(tree_features)
    
    # 如果未指定层级，则修正所有层级
    if levels_to_correct is None:
        levels_to_correct = list(tree_features.keys())
    
    for level_name in levels_to_correct:
        if level_name not in tree_features:
            print(f"Warning: Level '{level_name}' not found in tree_features. Skipping.")
            continue
        
        level_data = tree_features[level_name]
        level_mean = mean_features.get(level_name)
        
        if level_mean is None:
            print(f"Warning: Mean feature for level '{level_name}' not found. Skipping bias correction for this level.")
            continue
        
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

# apply_granularity_bias_correction 这种方式不行精度很低  

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


def correct_domain_bias_iNat(tree_features, mean_features, global_mean=None, layer_policy=None):
    """
    校正不同层级的文本特征的领域偏置。
    
    Args:
        tree_features (dict): 粒度层级特征字典，每个层级包含多个文本特征，形状为 {layer_name: {id: Tensor}}。
        mean_features (dict): 每个层级的均值特征，形状为 {layer_name: Tensor}。
        global_mean (torch.Tensor, optional): 所有领域的总体均值特征，形状为 [C]。
        layer_policy (dict): 每个层级的策略，例如 {'l1': 'no_gm', 'l2': 'gm'}。
    
    Returns:
        corrected_tree_features (dict): 校正后的粒度层级特征，形状与 tree_features 相同。
    """
    from copy import deepcopy
    corrected_tree_features = deepcopy(tree_features)
    
    for layer, features in tree_features.items():
        policy = layer_policy.get(layer, 'no_gm')  # 默认策略为 'no_gm'
        mean_feature = mean_features.get(layer)
        if mean_feature is None:
            raise ValueError(f"Mean feature for layer {layer} is missing.")
        
        for unique_id, text_features in features.items():
            # 校正领域偏置
            # 参数验证
            if not isinstance(policy, str):
                raise TypeError(f"policy should be a str, but got {type(policy)}")
            if policy not in ['gm', 'no_gm']:
                raise ValueError(f"policy should be 'gm' or 'no_gm', but got {policy}")
            
            # 检查输入的形状是否符合预期
            if text_features.dim() == 1:  # 如果是 [C]
                text_features = text_features.unsqueeze(0)  # 转换为 [1, C]
            elif text_features.dim() != 2:
                raise ValueError(f"text_features shape is invalid: {text_features.shape}, expected [C] or [N, C]")
            
            if mean_feature.dim() != 1:
                raise ValueError(f"mean_features shape is invalid: {mean_feature.shape}, expected [C]")
    
            # 计算领域偏置向量
            t_hat = mean_feature.clone()
            
            # 根据 policy 决定是否减去 global_mean
            if policy == 'gm' and global_mean is not None:
                t_hat -= global_mean  # 减去所有领域的总体均值
                if isinstance(text_features, list):
                    corrected_features = []
                    for feat in text_features:
                        corrected_feat = feat - t_hat
                        corrected_feat = corrected_feat / torch.norm(corrected_feat, p=2)
                        corrected_features.append(corrected_feat)
                    corrected_tree_features[level_name][unique_id] = corrected_features
                else:
                    corrected_feat = text_features - t_hat
                    corrected_feat = corrected_feat / torch.norm(corrected_feat, p=2)
                    if corrected_feat.shape[0] == 1:
                        corrected_feat = corrected_feat.squeeze(0)
                    corrected_tree_features[level_name][unique_id] = corrected_feat
            else:
                # 确保 t_hat 的形状可以广播到 text_features
                t_hat = t_hat.view(1, -1)  # [1, C]
        
                # 从文本特征中移除领域偏置
                centered_text = text_features - t_hat  # [N, C]
        
                # 计算 L2 范数
                norm = torch.norm(centered_text, p=2, dim=1, keepdim=True)  # [N, 1]
        
                # 避免除零并归一化
                corrected_text_features = centered_text / (norm + 1e-6)  # [N, C]
        
                # 如果原始输入是 [C]，返回时再转回 [C]
                if corrected_text_features.shape[0] == 1:
                    corrected_text_features = corrected_text_features.squeeze(0)
                
            # if policy == 'gm' and global_mean is not None:
            #     corrected_features = corrected_feat
            # else:
            #     corrected_features = corrected_text_features
                corrected_tree_features[layer][unique_id] = corrected_text_features
    
    return corrected_tree_features


# def correct_domain_bias_iNat0111(tree_features, mean_features, global_mean=None, layer_policy=None):
#     """
#     校正不同层级的文本特征的领域偏置。
    
#     Args:
#         tree_features (dict): 粒度层级特征字典，每个层级包含多个文本特征，形状为 {layer_name: {id: Tensor}}。
#         mean_features (dict): 每个层级的均值特征，形状为 {layer_name: Tensor}。
#         global_mean (torch.Tensor, optional): 所有领域的总体均值特征，形状为 [C]。
#         layer_policy (dict): 每个层级的策略，例如 {'l1': 'no_gm', 'l2': 'gm'}。
    
#     Returns:
#         corrected_tree_features (dict): 校正后的粒度层级特征，形状与 tree_features 相同。
#     """
#     from copy import deepcopy
#     corrected_tree_features = deepcopy(tree_features)
    
#     for layer, features in tree_features.items():
#         policy = layer_policy.get(layer, 'no_gm')  # 默认策略为 'no_gm'
#         mean_feature = mean_features.get(layer)
#         if mean_feature is None:
#             raise ValueError(f"Mean feature for layer {layer} is missing.")
        
#         for unique_id, text_features in features.items():
#             # 校正领域偏置
#             # 参数验证
#             if not isinstance(policy, str):
#                 raise TypeError(f"policy should be a str, but got {type(policy)}")
#             if policy not in ['gm', 'no_gm']:
#                 raise ValueError(f"policy should be 'gm' or 'no_gm', but got {policy}")
            
#             # 检查输入的形状是否符合预期
#             if text_features.dim() == 1:  # 如果是 [C]
#                 text_features = text_features.unsqueeze(0)  # 转换为 [1, C]
#             elif text_features.dim() != 2:
#                 raise ValueError(f"text_features shape is invalid: {text_features.shape}, expected [C] or [N, C]")
            
#             if mean_feature.dim() != 1:
#                 raise ValueError(f"mean_features shape is invalid: {mean_feature.shape}, expected [C]")
    
#             # 计算领域偏置向量
#             t_hat = mean_feature.clone()
            
#             # 根据 policy 决定是否减去 global_mean
#             if policy == 'gm' and global_mean is not None:
#                 t_hat -= global_mean  # 减去所有领域的总体均值
#                 corrected_feat = text_features - t_hat
#                 corrected_text_features = corrected_feat / torch.norm(corrected_feat, p=2)
#             else:
#                 # 确保 t_hat 的形状可以广播到 text_features
#                 t_hat = t_hat.view(1, -1)  # [1, C]
        
#                 # 从文本特征中移除领域偏置
#                 centered_text = text_features - t_hat  # [N, C]
        
#                 # 计算 L2 范数
#                 norm = torch.norm(centered_text, p=2, dim=1, keepdim=True)  # [N, 1]
        
#                 # 避免除零并归一化
#                 corrected_text_features = centered_text / (norm + 1e-6)  # [N, C]
        
#                 # 如果原始输入是 [C]，返回时再转回 [C]
#                 if corrected_text_features.shape[0] == 1:
#                     corrected_text_features = corrected_text_features.squeeze(0)
                
#             # if policy == 'gm' and global_mean is not None:
#             #     corrected_features = corrected_feat
#             # else:
#             #     corrected_features = corrected_text_features
#             corrected_tree_features[layer][unique_id] = corrected_text_features
    
#     return corrected_tree_features


import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv

def optimize_feature_with_adm_improved(
    current_feature,
    node_name_feature,
    node_name,
    cat_id,
    level_name,
    csv_writer=None,
    z=None,
    u=None,
    lambda_val=0.1,     # L1 强度
    beta=0.001,         # L2 强度 (Elastic Net)
    rho=1.0,
    num_epochs=100,
    lr=0.01,
    early_stop_patience=10
):
    """
    在原先的 ADMM 基础上，(1)加入L2正则(Elastic Net)，(2)自适应调节rho避免过度稀疏化。
    """
    # 初始化变量
    if z is None:
        z = torch.zeros_like(current_feature)
    if u is None:
        u = torch.zeros_like(current_feature)

    x = current_feature.clone().detach().requires_grad_(True)

    # 确保形状一致
    assert x.dim() == 1, f"x should be 1D tensor, got {x.dim()}D"
    assert x.shape == z.shape == u.shape, "Shapes of x, z, and u must be identical."

    print(f"Starting optimize_feature_with_adm_improved with lambda_val={lambda_val}, beta={beta}, rho={rho}")

    optimizer = optim.Adam([x], lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # 组件拆分
        # L2正则（beta * ||z||^2）
        loss_x = 0.5 * torch.norm(x - node_name_feature, p=2)**2
        loss_l1 = lambda_val * torch.norm(z, p=1)
        loss_l2 = beta * 0.5 * torch.norm(z, p=2)**2  # 新增Elastic Net
        loss_rho = (rho / 2) * torch.norm(x - z + u, p=2)**2

        loss = loss_x + loss_l1 + loss_l2 + loss_rho

        # 反向传播
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # 保存 z 的前一个状态
            z_prev = z.clone()

            # ========== 改进的 z 更新(考虑Elastic Net) ==========
            # Step1: normal ADMM soft-threshold w.r.t l1
            w = x + u
            alpha = lambda_val / rho
            z_tmp = torch.sign(w) * torch.clamp(torch.abs(w) - alpha, min=0)

            # Step2: l2 shrinkage
            gamma = beta / rho + 1e-12
            z = z_tmp / (1 + gamma)  # 简易写法

            # 更新对偶变量 u
            u = u + x - z

            # 打印 soft-thresholding 相关信息
            print(f"  [Debug] Epoch {epoch+1}: alpha={alpha:.4f}, max|x+u|={w.abs().max().item():.4f}")

        primal_residual = torch.norm(x - z, p=2).item()
        dual_residual = rho * torch.norm(z - z_prev, p=2).item()

        # 计算稀疏度
        nnz_ratio = (z.abs() > 1e-8).float().mean().item()

        # log
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch[{epoch+1}/{num_epochs}] Loss={loss.item():.4f},"
                  f" Lx={loss_x.item():.4f}, L1={loss_l1.item():.4f}, L2={loss_l2.item():.4f}, Lrho={loss_rho.item():.4f},"
                  f" Primal={primal_residual:.4f}, Dual={dual_residual:.4f}, lr={cur_lr}, rho={rho}")
            
            # 打印 z 和 u 的统计信息
            z_abs = z.abs()
            print(f"  [Debug] z stats: mean={z_abs.mean().item():.4f}, max={z_abs.max().item():.4f}, nonzero_ratio={nnz_ratio:.4f}")
            print(f"  [Debug] u stats: mean={u.abs().mean().item():.4f}, max={u.abs().max().item():.4f}")

            # CSV日志
            if csv_writer is not None:
                csv_writer.writerow([
                    node_name, cat_id, level_name,
                    loss.item(), primal_residual, dual_residual,
                    epoch+1, cur_lr
                ])

        # 自适应调节rho (防止过度稀疏)
        if nnz_ratio < 0.1:
            # 过度稀疏 -> 适当增大rho 或减小lambda_val
            rho *= 1.2
            print(f"  [Debug] Increasing rho to {rho} due to over-sparsity (nonzero ratio={nnz_ratio:.2f})")
            # 可选：lambda_val *= 0.95
        elif nnz_ratio > 0.8:
            # 过度密集 -> 可减小rho
            rho *= 0.9
            print(f"  [Debug] Decreasing rho to {rho} due to under-sparsity (nonzero ratio={nnz_ratio:.2f})")

        # 学习率调度
        scheduler.step(loss)

        # early stop
        if loss.item() < best_loss:
            best_loss = loss.item()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    return z.detach()
    # return z.detach(), u.detach()

def get_node_name_feature(node_name, global_encoder, device):
    """
    获取 node_name 的 CLIP 文本特征。
    """
    with torch.no_grad():
        tokens = clip.tokenize([node_name]).to(device)  # 单句
        node_feature = global_encoder.encode_text(tokens).float()  # [1, feature_dim]
    return node_feature.squeeze(0)  # [feature_dim]   

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

import argparse

def parse_k(value):
    """自定义的类型转换函数，将 'all' 转换为 None，否则转换为整数"""
    if value == 'all':
        return 'all'  # 返回 'all' 表示选择所有特征
    try:
        return int(value)  # 如果是数字，转换为整数
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid value for --k, must be 'all' or an integer")

def optimize_feature_with_adm_improved2(
    current_feature, 
    node_name_feature, 
    node_name,
    cat_id,
    level_name,
    csv_writer,
    lambda_val=0.1, 
    beta=0.001, 
    rho=1.0, 
    num_epochs=100, 
    lr=0.001, 
    early_stop_patience=10
):
    """
    使用 ADMM 优化 current_feature 使其逼近 node_name_feature，同时引入 L1 正则化。
    并将优化过程记录到 CSV 文件中。

    参数：
        current_feature (torch.Tensor): 当前特征，形状 [C]。
        node_name_feature (torch.Tensor): 节点名称特征，形状 [C]。
        node_name (str): 节点名称。
        cat_id (str/int): 分类 ID。
        level_name (str): 层级名称。
        csv_writer (csv.writer or None): CSV 写入器对象。如果为 None，则不记录日志。
        lambda_val (float): L1 正则化强度。
        beta (float): 未使用，保留以兼容函数签名。
        rho (float): ADMM 参数。
        num_epochs (int): 优化的最大 epoch 数。
        lr (float): 学习率。
        early_stop_patience (int): 早停的耐心值。

    返回：
        torch.Tensor: 优化后的特征张量。
    """
    device = current_feature.device
    x = current_feature.clone().detach().requires_grad_(True).to(device)
    z = torch.zeros_like(x).to(device)
    u = torch.zeros_like(x).to(device)

    # 设置优化器
    optimizer = optim.Adam([x], lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # 增广拉格朗日损失
        loss_reconstruction = 0.5 * torch.norm(x - node_name_feature, p=2)**2
        loss_l1 = lambda_val * torch.norm(z, p=1)
        loss_admm = (rho / 2) * torch.norm(x - z + u, p=2)**2
        loss = loss_reconstruction + loss_l1 + loss_admm

        # 检查损失是否为 NaN 或 Inf
        if not torch.isfinite(loss):
            print(f"[Error] Loss is {loss.item()} at epoch {epoch+1}. Aborting optimization.")
            return z.detach()

        # 反向传播
        loss.backward()

        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_([x], max_norm=1.0)

        # 更新权重
        optimizer.step()

        with torch.no_grad():
            # 保存 z 的前一个状态
            z_prev = z.clone()

            # 更新 z 使用软阈值化
            z = torch.sign(x + u) * torch.clamp(torch.abs(x + u) - lambda_val / rho, min=0)

            # 更新对偶变量 u
            u = u + x - z

            # 检查 x、z 和 u 是否为 NaN 或 Inf
            if not torch.isfinite(x).all():
                print(f"[Error] x contains NaN or Inf at epoch {epoch+1}. Aborting optimization.")
                return z.detach()
            if not torch.isfinite(z).all():
                print(f"[Error] z contains NaN or Inf at epoch {epoch+1}. Aborting optimization.")
                return z.detach()
            if not torch.isfinite(u).all():
                print(f"[Error] u contains NaN or Inf at epoch {epoch+1}. Aborting optimization.")
                return z.detach()

        # 计算残差
        primal_residual = torch.norm(x - z, p=2).item()
        dual_residual = torch.norm(-rho * (z - z_prev), p=2).item()

        # 记录日志
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Primal Residual: {primal_residual:.4f}, Dual Residual: {dual_residual:.4f}")
            print(f"x shape: {x.shape}, z shape: {z.shape}, u shape: {u.shape}")

            if csv_writer is not None:
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

def count_nan_inf(tensor, name="Tensor"):
    num_nans = torch.isnan(tensor).sum().item()
    num_infs = torch.isinf(tensor).sum().item()
    print(f"{name} - NaNs: {num_nans}, Infs: {num_infs}")


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='inat', choices=['inat', 'fsod'])
    parser.add_argument('--gpt_results_root', default='./UnSec/inat_llm_detail_answers')
    parser.add_argument('--prompter', default='isa', choices=['a', 'avg', 'concat', 'isa'])
    parser.add_argument('--aggregator', default='mean', choices=['peigen', 'mean', 'mixed', 'all_eigens'])
    parser.add_argument('--peigen_thresh', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--out_path', default='./nexus/inat/vitB32/UnSec_llm_by_level_TFC_0113_new_layer_policy_5n1g_new_epoch2_dgm') # UnSec_llm_by_level_TFC_0109_new_layer_policy_5n1g,UnSec_llm_by_level_TFC_0109_new_layer_policy_6g，UnSec_llm_by_level_TFC_0111_new_layer_policy_5n1g_w_SR_epoch100
    parser.add_argument('--clip_model', default="ViT-B/32", choices=['ViT-B/32', 'RN50'])
    parser.add_argument('--sentence_type', default='by_level', choices=['by_level', 'candidate', 'detail', 'combined'])
    parser.add_argument('--enable_global_mean', action='store_true', default=True, help='是否开启多领域总体均值校正') # 使用原本的修正方式前5层不需要开启gm第6层需要,使用最新的一直需要开启
    parser.add_argument('--enable_cross_level_mean', action='store_true', default=True, help='是否开启跨粒度层级的均值校正')
    parser.add_argument('--num_epochs', type=int, default=2, help="Number of epochs for optimization")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for optimization")
    parser.add_argument('--k', type=parse_k, default='all', help="Number of most relevant features to select, or 'all' to select all features")
    parser.add_argument('--optimizer', type=str, default='adm', choices=['orin', 'adm'], help="Select optimizer: 'adam' for traditional gradient descent, 'adm' for ADM")
    
    inat_layer_policy = {
    'l1': 'no_gm',
    'l2': 'no_gm',
    'l3': 'no_gm',
    'l4': 'no_gm',
    'l5': 'no_gm',
    'l6': 'gm'
    }

    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    if not is_valid_folder(args.out_path):
        raise FileExistsError

    # Device Selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset_name == 'inat':
        # level_names = ['l6','l5', 'l4', 'l3', 'l2', 'l1']
        level_names = ['l1']
    else:
        level_names = ['l3', 'l2', 'l1']
        # level_names = ['l1', 'l2', 'l3']

    print('Loading CLIP')  # 加载 CLIP 模型
    global_encoder, global_preprocess = clip.load(args.clip_model, device=device)
    theme_maker = Themer(method=args.aggregator, thresh=args.peigen_thresh, alpha=args.alpha)

    # 初始化 TextMeanFeatureCalculator
    # text_calculator = TextMeanFeatureCalculator2(save_dir='./MeanFeatureCalculator/CLip/inat/RN50/detail', device=device)

    text_calculator = TextMeanFeatureCalculator2(save_dir=args.out_path, device=device)

    # 收集文本特征
    parent_map, child_map = {}, {}
    theme_tree_features = defaultdict(dict)
    all_text_features = defaultdict(list)

   # 创建嵌套的类别名称到ID的映射
    category_name_to_id = defaultdict(dict)  # {level_name: {category_name: unique_id}}
    category_id_to_name = defaultdict(dict)  # {level_name: {unique_id: category_name}}

    for level_name in level_names:
        gpt_results_path = os.path.join(args.gpt_results_root, f"cleaned_{args.dataset_name}_gpt_detail_hrchy_{level_name}.json")
        if not os.path.exists(gpt_results_path):
            print(f"GPT results not found for level {level_name} at path {gpt_results_path}")
            continue

        gpt_results = load_json(gpt_results_path)
        for cat_id, entry in gpt_results.items():
            node_name = entry.get("node_name", None)  # 使用 "node_name" 而不是 "category_name"
            if node_name:
                normalized_name = node_name.lower()  # 统一转换为小写
                # unique_id = f"{level_name}_{cat_id}"  # 确保ID唯一，例如 "l1_1", "l2_2"
                unique_id = f"{cat_id}"  # 确保ID唯一，例如 "l1_1", "l2_2"

            candidate_sentences = entry.get("candidate_sentences", [])
            detail_sentences = entry.get("detail_sentences", [])
            sentences_to_use = select_sentences_by_level(candidate_sentences, detail_sentences, level_name, sentence_type=args.sentence_type)
            # 截断句子长度
            truncated_sentences = [sentence[:77] for sentence in sentences_to_use]
            node_tokens = clip.tokenize(truncated_sentences).to(device)
            with torch.no_grad():
                node_features = global_encoder.encode_text(node_tokens)

            # 生成当前节点的特征
            # current_feature = generate_features(global_encoder, truncated_sentences, device, aggregation='mean')
            # node_name_feature = get_node_name_feature(node_name, global_encoder, device)
            # node_theme = theme_maker.get_theme(node_features)
            # 生成当前节点的特征，并确保是独立副本
            current_feature = generate_features(global_encoder, truncated_sentences, device, aggregation='mean').clone().detach().to(torch.float32)
            node_theme = theme_maker.get_theme(node_features).clone().detach().to(torch.float32)
            node_name_feature = get_node_name_feature(node_name, global_encoder, device).clone().detach().to(torch.float32)  # 已经是 float32

            # 在优化前调用
            count_nan_inf(current_feature, "current_feature")
            count_nan_inf(node_theme, "node_theme")

            if args.optimizer == 'adm':
                print("当前优化层级：",level_name)
                print("当前优化节点：",node_name)
                optimized_feature = optimize_feature_with_adm_improved(##使用optimize_feature_with_adm_improved2效果很差
                    current_feature=node_theme,
                    node_name_feature=node_name_feature,
                    node_name=node_name,
                    cat_id=cat_id,
                    level_name=level_name,
                    csv_writer=None,
                    lambda_val=0.1,
                    beta=0.001,
                    rho=1.0,
                    num_epochs=args.num_epochs,
                    lr=args.lr,
                    early_stop_patience=10
                )
            # 收集特征用于均值计算
            all_text_features[level_name].append(optimized_feature)
            theme_tree_features[level_name][unique_id] = optimized_feature

    # mean_features = text_calculator.compute_mean_features_2(args.dataset_name, all_text_features, sentence_type=args.sentence_type)
    mean_features = compute_mean_features(theme_tree_features, save_dir=args.out_path, dataset_name=args.dataset_name, device=device)
    global_mean_features = None

    # Step 2: Compute global mean (μ_avg)
    global_mean = compute_global_mean(theme_tree_features)
    
    # corrected_tree_features = correct_domain_bias_iNat0111(theme_tree_features, mean_features, global_mean, inat_layer_policy)
    
    corrected_tree_features = correct_domain_bias_iNat(theme_tree_features, mean_features, global_mean, inat_layer_policy)

    for level_name, level_ids in corrected_tree_features.items():
        total_num = len(list(level_ids.values()))
        print(f"Total feats = {total_num} at {level_name}")

    # 打印和保存结果
    for level_name, level_theme_dict in corrected_tree_features.items():
        total_num = len(list(level_theme_dict.values()))
        print(f"Total feats = {total_num} at {level_name}")

        sorted_theme_dict = OrderedDict(sorted(level_theme_dict.items(), key=lambda x: int(x[0])))
        l_feats = list(sorted_theme_dict.values())
        l_classifier = torch.stack(l_feats)
        print(f"---> {level_name}'s classifier has a shape of {l_classifier.shape}")

        # 保存文本特征
        path_save = os.path.join(args.out_path, f"{args.dataset_name}_clip_hrchy_{level_name}.npy")
        print(f'Saving to {path_save}')
        np.save(open(path_save, 'wb'), l_classifier.cpu().numpy())


    # corrected_tree_features = correct_domain_bias_iNat(theme_tree_features, mean_features, global_mean, inat_layer_policy)

    # corrected_tree_features = apply_layer_specific_bias_correction(theme_tree_features, mean_features, global_mean, inat_layer_policy)  # UnSec_llm_by_level_TFC_0108_2
    # 修正领域偏置 UnSec_llm_by_level_TFC_0108_orin(当使用SR的时候要使用下面correct_domain_bias这种形式，不使用的时候 直接使用apply_layer_specific_bias_correction和inat_layer_policy)
    # for level_name, level_data in theme_tree_features.items():
    #     for cat_id, text_feature in level_data.items():
    #         if level_name in mean_features:
    #             domain_mean = mean_features[level_name]  # 获取当前粒度层级均值

    #             # 是否计算跨粒度的总体均值
    #             if args.enable_cross_level_mean:
    #                 cross_level_mean = torch.stack(list(mean_features.values())).mean(dim=0)
    #             else:
    #                 cross_level_mean = None

    #             # 是否计算所有领域的总体均值
    #             if args.enable_global_mean:
    #                 # 假设 global_mean_features 是提前预处理好的所有领域的均值
    #                 domain_invariant_mean = global_mean
    #             else:
    #                 domain_invariant_mean = None

    #             # 判断是否为第6层（假设第6层的 level_name 为 'level6'，根据实际情况调整）
    #             if level_name == 'l6':
    #                 print("修正领域偏置层级",level_name)
    #                 if isinstance(text_feature, list):
    #                     corrected_features = []
    #                     for feat in text_feature:
    #                         # 假设 global_mean 已经定义并可用
    #                         corrected_feat = feat - (domain_mean - global_mean)
    #                         corrected_feat = corrected_feat / torch.norm(corrected_feat, p=2)
    #                         corrected_features.append(corrected_feat)
    #                     theme_tree_features[level_name][cat_id] = corrected_features
    #                 else:
    #                     # 如果 text_feature 不是列表，按需处理
    #                     corrected_feat = text_feature - (domain_mean - global_mean)
    #                     corrected_feat = corrected_feat / torch.norm(corrected_feat, p=2)
    #                     theme_tree_features[level_name][cat_id] = corrected_feat
    #             else:
    #                 # 对其他层使用原有的修正方法
    #                 print("修正领域偏置层级",level_name)

    #                 # corrected_features = text_calculator.correct_domain_bias(
    #                 #     text_feature, domain_mean, cross_level_mean, domain_invariant_mean
    #                 # )

    #                 corrected_features = text_calculator.correct_domain_bias_iNat(
    #                     text_feature, domain_mean, cross_level_mean, global_mean,inat_layer_policy
    #                 )

    #                 theme_tree_features[level_name][cat_id] = corrected_features

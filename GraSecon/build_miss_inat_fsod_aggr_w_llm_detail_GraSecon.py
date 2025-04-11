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
                    corrected_tree_features[layer][unique_id] = corrected_features
                else:
                    corrected_feat = text_features - t_hat
                    corrected_feat = corrected_feat / torch.norm(corrected_feat, p=2)
                    if corrected_feat.shape[0] == 1:
                        corrected_feat = corrected_feat.squeeze(0)
                    corrected_tree_features[layer][unique_id] = corrected_feat
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

def visualize_all_granularity_levels_combined_3D(theme_tree_features_before, theme_tree_features_after, level_names, dataset_name, save_path=None, sample_per_level=None):
    """
    在单个3D空间中可视化所有粒度级别的特征分布对比，采用专业科研风格。
    
    Args:
        theme_tree_features_before (dict): 原始特征字典 (SHiNe)
        theme_tree_features_after (dict): 校正后特征字典 (Ours)
        level_names (list): 粒度级别名称列表
        dataset_name (str): 数据集名称
        save_path (str): 保存图像的路径
        sample_per_level (int): 每个级别采样的最大点数，用于大数据集
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import numpy as np
    import torch
    from matplotlib.lines import Line2D
    import os
    import matplotlib as mpl
    
    # 设置科研风格参数
    mpl.style.use('seaborn-v0_8-paper')
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = 0.3
    
    # 确保保存目录存在
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 定义专业科研风格的颜色方案
    scientific_colors = [
        '#3366CC', '#DC3912', '#109618', '#990099', '#0099C6', '#DD4477',
        '#66AA00', '#B82E2E', '#316395', '#994499', '#22AA99', '#AAAA11'
    ]
    
    # 创建图形
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # 收集所有级别的特征
    all_features_before = []
    all_features_after = []
    feature_level_mapping = []  # 记录每个特征点对应的粒度级别
    
    for i, level_name in enumerate(level_names):
        # 获取当前级别的特征
        if level_name in theme_tree_features_before and level_name in theme_tree_features_after:
            level_features_before = [f.cpu().numpy() if isinstance(f, torch.Tensor) else f 
                                    for f in theme_tree_features_before[level_name].values()]
            level_features_after = [f.cpu().numpy() if isinstance(f, torch.Tensor) else f 
                                   for f in theme_tree_features_after[level_name].values()]
            
            # 确保每个级别有相同数量的原始和校正后的特征
            min_count = min(len(level_features_before), len(level_features_after))
            if min_count == 0:
                print(f"Warning: No features found for level {level_name}. Skipping.")
                continue
            
            # 如果指定了采样数，则对大量点进行采样
            if sample_per_level and min_count > sample_per_level:
                import random
                # 随机选择索引
                selected_indices = random.sample(range(min_count), sample_per_level)
                
                level_features_before = [level_features_before[idx] for idx in selected_indices]
                level_features_after = [level_features_after[idx] for idx in selected_indices]
                min_count = sample_per_level
                print(f"Sampled {sample_per_level} points from level {level_name}")
            else:
                level_features_before = level_features_before[:min_count]
                level_features_after = level_features_after[:min_count]
            
            # 加入总体特征集合
            all_features_before.extend(level_features_before)
            all_features_after.extend(level_features_after)
            
            # 记录每个特征对应的粒度级别
            feature_level_mapping.extend([i] * min_count)
    
    # 将特征转换为numpy数组
    all_features_before = np.array(all_features_before)
    all_features_after = np.array(all_features_after)
    feature_level_mapping = np.array(feature_level_mapping)
    
    # 确保有足够的特征点进行PCA
    if len(all_features_before) < 3 or len(all_features_after) < 3:
        print("Error: Not enough features for PCA. Need at least 3 points.")
        return
    
    # 合并所有特征用于PCA计算
    combined_features = np.vstack((all_features_before, all_features_after))
    
    # 应用PCA降维
    pca = PCA(n_components=3)
    combined_reduced = pca.fit_transform(combined_features)
    
    # 分离降维后的原始特征和校正后的特征
    n_before = len(all_features_before)
    reduced_before = combined_reduced[:n_before]
    reduced_after = combined_reduced[n_before:]
    
    # 显示解释的方差比例
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by principal components: {explained_variance}")
    
    # 为每个粒度级别绘制散点图
    for i, level_name in enumerate(level_names):
        level_color = scientific_colors[i % len(scientific_colors)]
        
        if i in feature_level_mapping:
            # 获取当前粒度级别的特征索引
            level_indices = np.where(feature_level_mapping == i)[0]
            
            # 绘制原始特征（圆形标记）
            ax.scatter(
                reduced_before[level_indices, 0], 
                reduced_before[level_indices, 1], 
                reduced_before[level_indices, 2],
                color=level_color, marker='o', s=100, alpha=0.7, 
                edgecolors='black', linewidth=0.5,
                label=f'L{level_name[-1]} (SHiNe)' if i == 0 else ""
            )
            
            # 绘制校正后的特征（X形标记）
            ax.scatter(
                reduced_after[level_indices, 0], 
                reduced_after[level_indices, 1], 
                reduced_after[level_indices, 2],
                color=level_color, marker='x', s=120, alpha=0.8, 
                linewidth=2,
                label=f'L{level_name[-1]} (Ours)' if i == 0 else ""
            )
    
   # 设置坐标轴 - 增加labelpad解决重叠问题
    ax.set_xlabel(f'PC 1 ({explained_variance[0]*100:.1f}%)', fontsize=25, labelpad=10)
    ax.set_ylabel(f'PC 2 ({explained_variance[1]*100:.1f}%)', fontsize=25, labelpad=10)
    ax.set_zlabel(f'PC 3 ({explained_variance[2]*100:.1f}%)', fontsize=25, labelpad=10)
    
    # 确保刻度标签可见，减小刻度字体
    ax.tick_params(axis='x', labelsize=18, pad=3)
    ax.tick_params(axis='y', labelsize=18, pad=3)
    ax.tick_params(axis='z', labelsize=18, pad=3)
    
    # 可以选择性地减少刻度数量，避免拥挤
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=5)
    ax.locator_params(axis='z', nbins=5)
    
    # 调整图形大小和视角，给标签腾出更多空间
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    ax.view_init(elev=25, azim=35)
    
    # 创建自定义图例
    # 方法图例
    method_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=15, markeredgecolor='black', label='SHiNe'),
        Line2D([0], [0], marker='x', color='gray', markersize=15, 
               markeredgewidth=2, label='Ours')
    ]
    
    # 粒度级别图例
    level_handles = []
    for i, level_name in enumerate(level_names):
        if i in feature_level_mapping:
            level_color = scientific_colors[i % len(scientific_colors)]
            level_handles.append(
                Line2D([0], [0], marker='o', color=level_color, markersize=15, 
                       label=f'Level {level_name[-1]}')
            )
    
    # 将图例放在图形外部右侧，避免与内容重叠
    method_legend = ax.legend(handles=method_handles, 
                            loc='upper right', 
                            bbox_to_anchor=(1.15, 1.0),
                            fontsize=25, 
                            frameon=True, 
                            title="Method",
                            title_fontsize=25)
    
    # 获取第一个图例并添加到图形中
    ax.add_artist(method_legend)
    
    # 为第二个图例添加位置在右侧下方
    level_legend = fig.legend(handles=level_handles, 
                            loc='center right', 
                            bbox_to_anchor=(1.15, 0.5),
                            fontsize=25, 
                            frameon=True, 
                            title="Granularity Level",
                            title_fontsize=25)
    
    # 设置图例边框
    method_legend.get_frame().set_linewidth(1.5)
    level_legend.get_frame().set_linewidth(1.5)
    
    # 调整图形布局，确保有足够空间显示全部内容
    plt.tight_layout()
    
    # 保存图像时使用更大的边距确保不裁剪
    if save_path:
        fig_name = f"{save_path}/{dataset_name}_all_granularity_combined_3D.png"
        plt.savefig(fig_name, dpi=600, bbox_inches='tight', pad_inches=0.2)
        print(f"Figure saved to {fig_name}")
        
        # 同时保存为SVG
        svg_path = f"{save_path}/{dataset_name}_all_granularity_combined_3D.svg"
        plt.savefig(svg_path, format='svg', bbox_inches='tight', pad_inches=0.2)
        print(f"SVG figure saved to {svg_path}")
    
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='inat', choices=['inat', 'fsod'])
    parser.add_argument('--gpt_results_root', default='./GraSecon/GraSecon/inat_llm_detail_answers')
    parser.add_argument('--prompter', default='isa', choices=['a', 'avg', 'concat', 'isa'])
    parser.add_argument('--aggregator', default='mean', choices=['peigen', 'mean', 'mixed', 'all_eigens'])
    parser.add_argument('--peigen_thresh', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--out_path', default='./GraSecon/nexus/inat/vitB32/shine_llm_by_level_TFC_0113_new_layer_policy_5n1g_new_epoch2_dgm_visual') # shine_llm_by_level_TFC_0109_new_layer_policy_5n1g,shine_llm_by_level_TFC_0109_new_layer_policy_6g，shine_llm_by_level_TFC_0111_new_layer_policy_5n1g_w_SR_epoch100
    parser.add_argument('--clip_model', default="ViT-B/32", choices=['ViT-B/32', 'RN50'])
    parser.add_argument('--sentence_type', default='by_level', choices=['by_level', 'candidate', 'detail', 'combined'])
    parser.add_argument('--enable_global_mean', action='store_true', default=True, help='是否开启多领域总体均值校正') # 使用原本的修正方式前5层不需要开启gm第6层需要,使用最新的一直需要开启
    parser.add_argument('--enable_cross_level_mean', action='store_true', default=True, help='是否开启跨粒度层级的均值校正')
    parser.add_argument('--num_epochs', type=int, default=2, help="Number of epochs for optimization")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for optimization")
    parser.add_argument('--k', type=parse_k, default='all', help="Number of most relevant features to select, or 'all' to select all features")
    parser.add_argument('--optimizer', type=str, default='orin', choices=['orin', 'adm'], help="Select optimizer: 'adam' for traditional gradient descent, 'adm' for ADM")
    
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
        level_names = ['l6','l5', 'l4', 'l3', 'l2', 'l1']
        # level_names = ['l6']
    else:
        level_names = ['l3', 'l2', 'l1']
        # level_names = ['l1', 'l2', 'l3']

    print('Loading CLIP')  # 加载 CLIP 模型
    global_encoder, global_preprocess = clip.load(args.clip_model, device=device)
    theme_maker = Themer(method=args.aggregator, thresh=args.peigen_thresh, alpha=args.alpha)

    # 初始化 TextMeanFeatureCalculator
    # text_calculator = TextMeanFeatureCalculator2(save_dir='./GraSecon/MeanFeatureCalculator/CLip/inat/RN50/detail', device=device)

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

            # 收集特征用于均值计算
            all_text_features[level_name].append(node_name_feature)
            theme_tree_features[level_name][unique_id] = node_name_feature

    # mean_features = text_calculator.compute_mean_features_2(args.dataset_name, all_text_features, sentence_type=args.sentence_type)
    mean_features = compute_mean_features(theme_tree_features, save_dir=args.out_path, dataset_name=args.dataset_name, device=device)
    global_mean_features = None

    # Step 2: Compute global mean (μ_avg)
    global_mean = compute_global_mean(theme_tree_features)
    
    # corrected_tree_features = correct_domain_bias_iNat0111(theme_tree_features, mean_features, global_mean, inat_layer_policy)
    
    corrected_tree_features = correct_domain_bias_iNat(theme_tree_features, mean_features, global_mean, inat_layer_policy)

    save_path = './GraSecon/visualize/PCA/inat/vitB32/visualize_all_granularity_levels_combined_3D'
    os.makedirs(save_path, exist_ok=True)

    visualize_all_granularity_levels_combined_3D(
        theme_tree_features, 
        corrected_tree_features, 
        level_names, 
        args.dataset_name, 
        save_path=save_path,
        sample_per_level=500
    )

    print(f"可视化已保存至: {save_path}")
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



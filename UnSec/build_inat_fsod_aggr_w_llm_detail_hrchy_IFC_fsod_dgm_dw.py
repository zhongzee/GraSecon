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
    all_exists = all(os.path.exists(path) for path in mean_feature_paths.values())
    if all_exists:
        for level_name, path in mean_feature_paths.items():
            mean_features[level_name] = torch.from_numpy(np.load(path)).to(device)
        print(f"已加载数据集 {dataset_name} 的所有层级均值特征")
        return mean_features

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

def apply_weighted_bias_correction(tree_features, mean_features, global_mean=None, layer_weights=None):
    """
    对不同层级应用加权的偏置修正策略。
    Args:
        tree_features (dict): 粒度层级特征字典。
        mean_features (dict): 每个粒度层级的均值特征。
        global_mean (Tensor): 全局均值特征。
        layer_weights (dict): 每个层级的权重，例如 {'l1': 0.0, 'l2': 0.5, 'l3': 1.0}。
    Returns:
        corrected_tree_features (dict): 修正后的粒度层级特征。
    """
    corrected_tree_features = deepcopy(tree_features)
    
    for level_name, level_data in tree_features.items():
        level_mean = mean_features[level_name]
        weight = layer_weights.get(level_name, 1.0)  # 默认权重为1.0
        for unique_id, feature in level_data.items():
            if isinstance(feature, list):
                corrected_features = []
                for feat in feature:
                    if global_mean is not None:
                        # corrected_feat = feat - weight * (level_mean - global_mean)
                        corrected_feat = feat - (level_mean - weight * global_mean)
                    else:
                        # corrected_feat = feat - weight * level_mean
                        corrected_feat = feat - level_mean
                    corrected_feat = corrected_feat / torch.norm(corrected_feat, p=2)
                    corrected_features.append(corrected_feat)
                corrected_tree_features[level_name][unique_id] = corrected_features
            else:
                if global_mean is not None:
                    corrected_feat = feature - (level_mean - weight * global_mean)
                else:
                    corrected_feat = feature - level_mean
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

import torch
from copy import deepcopy

def layer_specific_adaptive_feature_standardization(feature, mu, sigma, gamma=1.0, beta=0.0):
    """
    对单个特征应用层级特定的自适应特征标准化。
    
    Args:
        feature (Tensor): 原始特征向量，形状为 [feature_dim]。
        mu (Tensor): 层级均值向量，形状为 [feature_dim]。
        sigma (Tensor): 层级标准差向量，形状为 [feature_dim]。
        gamma (float): 缩放因子。
        beta (float): 偏移因子。
    
    Returns:
        Tensor: 标准化后的特征向量，形状为 [feature_dim]。
    """
    standardized_feat = (feature - mu) / sigma
    standardized_feat = standardized_feat * gamma + beta
    standardized_feat = standardized_feat / torch.norm(standardized_feat, p=2)
    return standardized_feat


def apply_layer_specific_self_bias_correction(tree_features, mean_features, global_mean=None, layer_policy=None):
    """
    对不同层级应用不同的偏置修正策略。
    
    Args:
        tree_features (dict): 粒度层级特征字典，结构为 {层级: {唯一ID: 特征向量或特征向量列表}}。
        mean_features (dict): 每个粒度层级的均值特征，结构为 {层级: 均值向量}。
        global_mean (Tensor, optional): 全局均值特征向量。
        layer_policy (dict, optional): 每个层级的策略，例如 {'l1': 'no_gm', 'l2': 'gm'}。
    
    Returns:
        dict: 修正后的粒度层级特征字典。
    """
    corrected_tree_features = deepcopy(tree_features)
    
    if layer_policy is None:
        layer_policy = {}
    
    # 识别所有 'no_gm' 层级
    no_gm_layers = [layer for layer, policy in layer_policy.items() if policy == 'no_gm']
    
    # 预计算 'no_gm' 层级的均值和标准差
    layer_stats = {}
    for layer in no_gm_layers:
        all_features = []
        for unique_id, feature in tree_features[layer].items():
            if isinstance(feature, list):
                all_features.extend(feature)
            else:
                all_features.append(feature)
        if len(all_features) == 0:
            raise ValueError(f"Layer '{layer}' has no features to standardize.")
        all_features_tensor = torch.stack(all_features)  # 形状: [num_instances, feature_dim]
        mu = all_features_tensor.mean(dim=0)
        sigma = all_features_tensor.std(dim=0) + 1e-5  # 避免除以零
        layer_stats[layer] = (mu, sigma)
    
    # 逐层应用偏置校正策略
    for level_name, level_data in tree_features.items():
        policy = layer_policy.get(level_name, 'gm')  # 默认策略为 'gm'
        
        if policy == 'gm':
            # 应用全局均值校正
            for unique_id, feature in level_data.items():
                if isinstance(feature, list):
                    corrected_features = []
                    for feat in feature:
                        if global_mean is not None:
                            corrected_feat = feat - (mean_features[level_name] - global_mean)
                        else:
                            corrected_feat = feat - mean_features[level_name]
                        corrected_feat = corrected_feat / torch.norm(corrected_feat, p=2)
                        corrected_features.append(corrected_feat)
                    corrected_tree_features[level_name][unique_id] = corrected_features
                else:
                    if global_mean is not None:
                        corrected_feat = feature - (mean_features[level_name] - global_mean)
                    else:
                        corrected_feat = feature - mean_features[level_name]
                    corrected_feat = corrected_feat / torch.norm(corrected_feat, p=2)
                    corrected_tree_features[level_name][unique_id] = corrected_feat
        
        elif policy == 'no_gm':
            # 应用层级特定的自适应特征标准化
            mu, sigma = layer_stats[level_name]
            for unique_id, feature in level_data.items():
                if isinstance(feature, list):
                    corrected_features = []
                    for feat in feature:
                        standardized_feat = layer_specific_adaptive_feature_standardization(feat, mu, sigma, gamma=1.0, beta=0.0)
                        corrected_features.append(standardized_feat)
                    corrected_tree_features[level_name][unique_id] = corrected_features
                else:
                    standardized_feat = layer_specific_adaptive_feature_standardization(feature, mu, sigma, gamma=1.0, beta=0.0)
                    corrected_tree_features[level_name][unique_id] = standardized_feat
        
        else:
            # 如果策略不是 'gm' 或 'no_gm'，默认应用 'gm'
            for unique_id, feature in level_data.items():
                if isinstance(feature, list):
                    corrected_features = []
                    for feat in feature:
                        corrected_feat = feat - mean_features[level_name]
                        corrected_feat = corrected_feat / torch.norm(corrected_feat, p=2)
                        corrected_features.append(corrected_feat)
                    corrected_tree_features[level_name][unique_id] = corrected_features
                else:
                    corrected_feat = feature - mean_features[level_name]
                    corrected_feat = corrected_feat / torch.norm(corrected_feat, p=2)
                    corrected_tree_features[level_name][unique_id] = corrected_feat
    
    return corrected_tree_features


# 定义层级权重
layer_weights = {
    'l1': 0,  # 不应用GM
    'l2': 1,  # 部分应用GM
    'l3': 1.2   # 全部应用GM
}

# 定义层级策略
# layer_policy = {
#     'l1': 'no_gm',
#     'l2': 'gm',
#     'l3': 'gm'
# }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='fsod', choices=['inat', 'fsod'])
    parser.add_argument('--gpt_results_root', default='./UnSec/fsod_llm_detail_answers')
    parser.add_argument('--prompter', default='isa', choices=['a', 'avg', 'concat', 'isa'])
    parser.add_argument('--aggregator', default='mean', choices=['peigen', 'mean', 'mixed', 'all_eigens'])
    parser.add_argument('--peigen_thresh', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--out_path', default='./nexus/fsod/vitB32/UnSec_llm_TFC_dgm_w_0_1_1.2_1231') # UnSec_llm_by_level_TFC_dgm
    parser.add_argument('--clip_model', default="ViT-B/32", choices=['ViT-B/32', 'RN50'])
    parser.add_argument('--sentence_type', default='candidate', choices=['by_level', 'candidate', 'detail', 'combined'])
    parser.add_argument('--enable_global_mean', action='store_true', default=True, help='是否开启多领域总体均值校正')
    parser.add_argument('--enable_cross_level_mean', action='store_true', default=True, help='是否开启跨粒度层级的均值校正')

    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    if not is_valid_folder(args.out_path):
        raise FileExistsError

    # Device Selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset_name == 'inat':
        level_names = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']
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
    theme_tree_features = defaultdict(dict)
    all_text_features = defaultdict(list)

   # 创建嵌套的类别名称到ID的映射

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
            # 截断句子长度 在这个地方 可以做一个优化！减少这个文本长度太长带来的问题
            truncated_sentences = [sentence[:77] for sentence in sentences_to_use]
            orin_tokens = clip.tokenize(sentences_to_use).to(device)
            node_tokens = clip.tokenize(truncated_sentences).to(device)
            with torch.no_grad():
                node_features = global_encoder.encode_text(node_tokens)

            with torch.no_grad():
                node_features = global_encoder.encode_text(orin_tokens)
                
            # 收集特征用于均值计算
            all_text_features[level_name].append(node_features)

            # 获取主题特征
            node_theme = theme_maker.get_theme(node_features)
            theme_tree_features[level_name][unique_id] = node_theme


    # Step 1: Compute mean (μ_i)
    # mean_features = text_calculator.compute_mean_features(args.dataset_name, all_text_features, sentence_type=args.sentence_type)
    mean_features = compute_mean_features(theme_tree_features, save_dir=args.out_path, dataset_name=args.dataset_name, device=device)
 
    global_mean_features = None

    # Step 2: Compute global mean (μ_avg)
    global_mean = compute_global_mean(theme_tree_features)
    
    # Step 3：动态低层次使用gm
    # corrected_tree_features = apply_layer_specific_bias_correction(theme_tree_features, mean_features, global_mean, layer_policy=layer_policy)

    # 第三层使用自身的偏置
    # corrected_tree_features = apply_layer_specific_self_bias_correction(theme_tree_features, mean_features, global_mean, layer_weights=layer_weights)

    # 应用加权偏置修正
    corrected_tree_features = apply_weighted_bias_correction(theme_tree_features, mean_features, global_mean, layer_weights=layer_weights)

    

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

    # corrected_tree_features = apply_granularity_bias_correction(theme_tree_features, mean_features, global_mean, device)

    # theme_tree_features = correct_domain_fsod2(theme_tree_features,parent_map,child_map,category_name_to_id,level_names)

    # 修正领域偏置
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
    #                 domain_invariant_mean = global_mean_features.get(level_name, None)
    #             else:
    #                 domain_invariant_mean = None

    #             # 修正领域偏置
    #             print("修正领域偏置")
    #             theme_tree_features[level_name][cat_id] = text_calculator.correct_domain_bias(
    #                 text_feature, domain_mean, cross_level_mean, domain_invariant_mean
    #             )



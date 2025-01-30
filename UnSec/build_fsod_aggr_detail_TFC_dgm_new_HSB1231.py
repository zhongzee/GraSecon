import argparse
import torch
import numpy as np
import clip
from collections import defaultdict, OrderedDict
from copy import deepcopy
from tools.composer import SignatureComposer
from tools.themer import Themer
from tools.fileios import *


def load_detail_sentences(hierarchy_dir, level_names):
    """
    Load detail_sentences for each level from JSON files into a dictionary.
    """
    detail_dict = {}
    for level in level_names:
        file_path = os.path.join(hierarchy_dir, f"cleaned_fsod_gpt_detail_hrchy_{level}.json")
        with open(file_path, 'r') as f:
            data = json.load(f)
            for node_id, content in data.items():
                detail_dict[content['node_name']] = content['detail_sentences']
    return detail_dict

def get_mean_vector(stacked_feats):
    print("========== used Mean when shape={}".format(stacked_feats.shape))
    mean_theme = torch.mean(stacked_feats, dim=0)  # mean vector
    return mean_theme

def compute_leaf_embedding(htree, leaf_level, composer, clip_model='ViT-B/32', detail_dict=None, device='cuda'):
    import torch
    import clip  # 确保已安装 openai/clip

    l_names = ['l3', 'l2', 'l1']

    meta_level_leaf = htree.get(leaf_level)

    # Extract class ids with their parents
    signature_ids = [[int(x['id'])] for x in sorted(meta_level_leaf['categories'], key=lambda x: x['id'])]
    for i in range(len(signature_ids)):
        leaf_id = str(signature_ids[i][0])
        parents_ids = meta_level_leaf['parents'].get(leaf_id)
        if parents_ids:
            signature_ids[i].extend(parents_ids)
        else:
            # 填充缺失的父级
            signature_ids[i].extend([0] * (len(l_names) - len(signature_ids[i])))

    signature_names = []
    for cat_id in signature_ids:
        cat_name = []
        for level_idx, this_id in enumerate(cat_id):
            if this_id == 0:
                cat_name.append("unknown")  # 或其他占位符
                continue
            level_name = l_names[level_idx]
            try:
                this_name = htree[level_name]['categories'][this_id - 1]['name']
            except IndexError:
                this_name = "unknown"
            cat_name.append(this_name)
        signature_names.append(cat_name)

    assert len(signature_ids) == len(signature_names)
    assert all(len(signature_id) == len(signature_name) for signature_id, signature_name in
               zip(signature_ids, signature_names))

    # Compose sentences from signatures
    sentences = composer.compose(signature_names)

    for sent in sentences:
        print("orin_sent", sent)

    # Match and append detail sentences to each signature
    detail_sentences = []
    for signature in signature_names:
        details = []
        for name in signature:
            if name in detail_dict:
                details.extend(detail_dict[name])
        # Concatenate detail sentences into one string
        detail_sentence = " ".join(details) if details else ""
        detail_sentences.append(detail_sentence)

    for sent in detail_sentences:
        print("detai_sent", sent)

    print('Loading CLIP')
    model, preprocess = clip.load(clip_model, device=device)

    # Encode sentences
    text = clip.tokenize(sentences).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)  # [200, 512]

    truncated_sentences = [sentence[:77] for sentence in detail_sentences]
    # Encode detail sentences
    detail_tokens = clip.tokenize(truncated_sentences).to(device)
    
    with torch.no_grad():
        detail_features = model.encode_text(detail_tokens)  # [200, 512]

    # Ensure detail features match text features length
    assert detail_features.shape[0] == text_features.shape[0], "Mismatch between sentences and detail_sentences encoding."

    # Combine features by averaging
    combined_features = (text_features + detail_features) / 2  # [200, 512]

    return combined_features


# def compute_leaf_embedding(htree, leaf_level, composer, clip_model='ViT-B/32', detail_dict=None, device='cuda'):
#     l_names = ['l3', 'l2', 'l1']

#     meta_level_leaf = htree.get(leaf_level)

#     # Extract class ids with their parents
#     signature_ids = [[int(x['id'])] for x in sorted(meta_level_leaf['categories'], key=lambda x: x['id'])]
#     for i in range(len(signature_ids)):
#         leaf_id = str(signature_ids[i][0])
#         parents_ids = meta_level_leaf['parents'].get(leaf_id)
#         signature_ids[i].extend(parents_ids)

#     signature_names = []
#     for cat_id in signature_ids:
#         cat_name = []
#         for level_idx, this_id in enumerate(cat_id):
#             level_name = l_names[level_idx]
#             this_name = htree[level_name]['categories'][this_id - 1]['name']
#             cat_name.append(this_name)
#         signature_names.append(cat_name)

#     assert len(signature_ids) == len(signature_names)
#     assert all(len(signature_id) == len(signature_name) for signature_id, signature_name in
#                zip(signature_ids, signature_names))

#     # Compose sentences from signatures
#     sentences = composer.compose(signature_names)

#     for sent in sentences:
#         print("orin_sent",sent)

#     # Match and append detail sentences to each signature
#     detail_sentences = []
#     for signature in signature_names:
#         details = []
#         for name in signature:
#             if name in detail_dict:
#                 details.extend(detail_dict[name])
#         detail_sentences.append(details)

#     for sent in detail_sentences:
#         print("detai_sent",sent)

#     print('Loading CLIP')
#     model, preprocess = clip.load(clip_model, device=device)

#     # Encode sentences
#     text = clip.tokenize(sentences).to(device)
#     with torch.no_grad():
#         text_features = model.encode_text(text)

#     # Encode detail sentences and combine
#     detail_features = []

#     detail_text = clip.tokenize(detail_sentences).to(device)
#     with torch.no_grad():
#         detail_features = model.encode_text(detail_text)
#     #     detail_features.append(avg_detail_feature)

#     # Ensure detail features match text features length
#     assert len(detail_features) == len(text_features), "Mismatch between sentences and detail_sentences encoding."

#     # Combine features
#     combined_features = []
#     for text_feat, detail_feat in zip(text_features, detail_features):
#         stacked_feats = torch.stack([text_feat, detail_feat])
#         combined = get_mean_vector(stacked_feats)  # Use _get_mean_vector
#         combined_features.append(combined)
#     # Return combined features
#     return torch.stack(combined_features)

    # # Encode detail_sentences and calculate mean
    # for detail_set in detail_sentences:
    #     if not detail_set:
    #         continue
    #     detail_tokens = clip.tokenize(detail_set).to(device)
    #     with torch.no_grad():
    #         encoded_details = model.encode_text(detail_tokens)
    #     avg_detail_feature = get_mean_vector(encoded_details)  # Use _get_mean_vector

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


def apply_granularity_bias_correction(tree_features, mean_features, global_mean = None):
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
                    if global_mean is not None:
                        corrected_feat = feat - (level_mean - global_mean)  # 去除粒度特定偏置，保留全局信息
                    else:
                        corrected_feat = feat - level_mean
                    corrected_feat = corrected_feat / torch.norm(corrected_feat, p=2)  # L2归一化
                    corrected_features.append(corrected_feat)
                corrected_tree_features[level_name][unique_id] = corrected_features
            else:
                # 单一特征的修正
                if global_mean is not None:
                    corrected_feat = feature - (level_mean - global_mean)  # 去除粒度特定偏置，保留全局信息
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

# 定义层级策略
layer_policy = {
    'l1': 'no_gm',
    'l2': 'gm',
    'l3': 'gm'
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', default='./UnSec/fsod_annotations/fsod_hierarchy_tree.json')
    parser.add_argument('--prompter', default='isa', choices=['a', 'concat', 'isa'])
    parser.add_argument('--aggregator', default='mean', choices=['peigen', 'mean', 'mixed', 'all_eigens'])
    parser.add_argument('--out_path', default='./nexus/fsod/vitB32/UnSec_GT_detail_TFC_nggm_rHSB_1231')
    parser.add_argument('--peigen_thresh', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--clip_model', default="ViT-B/32", choices=['ViT-B/32', 'RN50'])
    parser.add_argument('--dataset_name', default='fsod', choices=['inat', 'fsod'])
    parser.add_argument('--enable_global_mean', action='store_true', default=True, help='是否开启多领域总体均值校正') # 需要

    args = parser.parse_args()

    import os

    if not os.path.exists(args.out_path):
        print(f"Output directory '{args.out_path}' does not exist. Creating it...")
        os.makedirs(args.out_path)


    if not is_valid_folder(args.out_path):
        raise FileExistsError

    # Device Selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print('Loading CLIP')
    global_encoder, global_preprocess = clip.load(args.clip_model, device=device)

    # Load Metadata
    print(f'Loading {args.tree}')
    level_names = ['l3', 'l2', 'l1']

    meta_tree = json.load(open(args.tree, 'r'))
    meta_level_leaf = meta_tree.get('l3')

    # Load detail sentences
    hierarchy_dir = "./UnSec/fsod_llm_detail_answers/"
    detail_dict = load_detail_sentences(hierarchy_dir, level_names)

    # Extract class ids with their parents
    signature_ids = [[int(x['id'])] for x in sorted(meta_level_leaf['categories'], key=lambda x: x['id'])]
    for i in range(len(signature_ids)):
        leaf_id = str(signature_ids[i][0])
        parents_ids = meta_level_leaf['parents'].get(leaf_id)
        signature_ids[i].extend(parents_ids)

    tree_childs_to_leaf = {
        'l1': {},
        'l2': {},
    }

    for leaf_signature in signature_ids:
        cat_id_at_leaf = leaf_signature[0]
        for level_idx, level_name in enumerate(level_names[1:], start=1):
            this_level_parent_id = str(leaf_signature[level_idx])
            if this_level_parent_id in tree_childs_to_leaf[level_name]:
                tree_childs_to_leaf[level_name][this_level_parent_id].append(cat_id_at_leaf)
            else:
                tree_childs_to_leaf[level_name][this_level_parent_id] = [cat_id_at_leaf]

    prompt_composer = SignatureComposer(prompter=args.prompter)

    # Compute Leaf Features
    leaf_features = compute_leaf_embedding(
        meta_tree, level_names[0], composer=prompt_composer,
        clip_model=args.clip_model, detail_dict=detail_dict,
        device=device
    )

    theme_maker = Themer(method=args.aggregator, thresh=args.peigen_thresh, alpha=args.alpha)

    print(leaf_features.shape)

    tree_features = defaultdict(dict)
    for i, feat in enumerate(leaf_features, start=1):
        tree_features[level_names[0]][str(i)] = feat

    for level_name, level_childs in tree_childs_to_leaf.items():
        for level_cat_id, level_child_ids in level_childs.items():
            print(f"{level_name} cat_{level_cat_id} has childs: {level_child_ids}")

            level_child_feats = [tree_features[level_names[0]][str(idx)]
                                 for idx in level_child_ids]
            tree_features[level_name][level_cat_id] = level_child_feats

    theme_tree_features = deepcopy(tree_features)

    for level_name, level_ids in tree_features.items():
        if level_name == 'l3':
            continue

        for level_cat_id, level_child_feats in level_ids.items():
            stacked_child_feats = torch.stack(level_child_feats)
            theme_feat = theme_maker.get_theme(stacked_child_feats)
            theme_tree_features[level_name][level_cat_id] = theme_feat

    
    #  TFC
    # Step 1: Compute mean features for each level
    mean_features = compute_mean_features(theme_tree_features, save_dir=args.out_path, dataset_name=args.dataset_name, device=device)

    # Step 2: Compute global mean (μ_avg)
    global_mean = None
    if args.enable_global_mean:
        global_mean = compute_global_mean(theme_tree_features)
    # Step 3: Apply domain bias correction
    # corrected_tree_features = apply_granularity_bias_correction(theme_tree_features, mean_features, global_mean)

    # 应用分层偏置修正
    corrected_tree_features = apply_layer_specific_bias_correction(theme_tree_features, mean_features, global_mean, layer_policy=layer_policy)


    for level_name, level_ids in corrected_tree_features.items():
        total_num = len(list(level_ids.values()))
        print(f"Total feats = {total_num} at {level_name}")

    # Prepare and Save Features
    for level_name, level_theme_dict in corrected_tree_features.items():
        sorted_theme_dict = OrderedDict(sorted(level_theme_dict.items(), key=lambda x: int(x[0])))

        l_feats = list(sorted_theme_dict.values())
        l_classifier = torch.stack(l_feats)
        print(f"---> {level_name}'s classifier has a shape of {l_classifier.shape}")
        # Save the embeddings
        path_save = os.path.join(args.out_path, f"fsod_clip_hrchy_{level_name}.npy")
        print(f'Saving to {path_save}')
        np.save(open(path_save, 'wb'), l_classifier.cpu().numpy())

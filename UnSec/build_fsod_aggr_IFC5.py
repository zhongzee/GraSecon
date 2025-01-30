import argparse
import torch
import numpy as np
import clip
from collections import defaultdict, OrderedDict
from copy import deepcopy
from tools.composer import SignatureComposer
from tools.themer import Themer
from tools.fileios import *
from tools.IFC2 import TextMeanFeatureCalculator2

def compute_leaf_embedding(htree, leaf_level, composer, clip_model='ViT-B/32', device='cuda'):
    l_names = ['l3', 'l2', 'l1']
    meta_level_leaf = htree.get(leaf_level)

    # extract class ids w/ its parents
    signature_ids = [[int(x['id'])] for x in sorted(meta_level_leaf['categories'], key=lambda x: x['id'])]
    for i in range(len(signature_ids)):
        leaf_id = str(signature_ids[i][0])
        parents_ids = meta_level_leaf['parents'].get(leaf_id)
        signature_ids[i].extend(parents_ids)

    signature_names = []
    for cat_id in signature_ids:
        cat_name = []
        for level_idx, this_id in enumerate(cat_id):
            level_name = l_names[level_idx]
            this_name = htree[level_name]['categories'][this_id-1]['name']
            cat_name.append(this_name)
        signature_names.append(cat_name)

    assert len(signature_ids) == len(signature_names)
    assert all(len(signature_id) == len(signature_name) for signature_id, signature_name in
               zip(signature_ids, signature_names))

    sentences = composer.compose(signature_names)
    for sent in sentences:
        print(sent)

    print('Loading CLIP')
    model, preprocess = clip.load(clip_model, device=device)

    # tokenize class names
    text = clip.tokenize(sentences).to(device)

    # encoding
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features#F.normalize(text_features)


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


# 这个UnSec_GT_by_level_IFC_gm3完全不行，应该是计算逻辑出错
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', default='./UnSec/fsod_annotations/fsod_hierarchy_tree.json')
    parser.add_argument('--prompter', default='isa', choices=['a', 'concat', 'isa'])
    parser.add_argument('--aggregator', default='mean', choices=['peigen', 'mean', 'mixed', 'all_eigens'])
    parser.add_argument('--out_path', default='./nexus/fsod/vitB32/UnSec_GT_by_level_IFC_gm3')

    parser.add_argument('--peigen_thresh', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--clip_model', default="ViT-B/32", choices=['ViT-B/32', 'RN50'])
    parser.add_argument('--dataset_name', default='fsod', choices=['inat', 'fsod'])
    parser.add_argument('--sentence_type', default='by_level', choices=['by_level', 'candidate', 'detail', 'combined'])
    parser.add_argument('--enable_global_mean', action='store_true', default=False, help='是否开启多领域总体均值校正')
    parser.add_argument('--enable_cross_level_mean', action='store_true', default=True, help='是否开启跨粒度层级的均值校正')
    
    args = parser.parse_args()
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    if not is_valid_folder(args.out_path): raise FileExistsError

    # Device Selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print('Loading CLIP')
    global_encoder, global_preprocess = clip.load(args.clip_model, device=device)

    # Load Metadata
    print(f'Loading {args.tree}')
    level_names = ['l3', 'l2', 'l1']

    meta_tree = json.load(open(args.tree, 'r'))
    meta_level_leaf = meta_tree.get('l3')

    # extract class ids w/ its parents
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
        meta_tree, level_names[0],
        composer=prompt_composer, clip_model=args.clip_model,
        device=device
    )

    theme_maker = Themer(method=args.aggregator, thresh=args.peigen_thresh, alpha=args.alpha)

    print(leaf_features.shape)
    # 通过子节点特征构建非叶子层级的特征字典
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

    # 聚合非叶子节点

    for level_name, level_ids in tree_features.items():
        if level_name == 'l3':
            continue

        for level_cat_id, level_child_feats in level_ids.items():
            stacked_child_feats = torch.stack(level_child_feats)
            theme_feat = theme_maker.get_theme(stacked_child_feats)
            theme_tree_features[level_name][level_cat_id] = theme_feat


    # Step 1: Compute mean features for each level
    mean_features = compute_mean_features(theme_tree_features, save_dir=args.out_path, dataset_name=args.dataset_name, device=device)

    text_calculator = TextMeanFeatureCalculator2(save_dir='./nexus/inat/vitB32/UnSec_GT_by_level_IFC_gm3', device=device)

    global_mean_features = None

    # Step 2: Compute global mean (\u03bc_avg) if enabled
    if args.enable_cross_level_mean:
        cross_level_mean = torch.stack(list(mean_features.values())).mean(dim=0)
    else:
        cross_level_mean = None

    # Step 3: Apply domain bias correction
    for level_name, level_data in theme_tree_features.items():
        for cat_id, text_feature in level_data.items():
            if level_name in mean_features:
                domain_mean = mean_features[level_name]  # Retrieve granularity-level mean

                # Determine whether to compute the domain-invariant mean
                if args.enable_global_mean:
                    # Assuming global_mean_features has been precomputed for all domains
                    domain_invariant_mean = global_mean_features.get(level_name, None)
                else:
                    domain_invariant_mean = None

                # Correct domain bias
                print("Applying domain bias correction")
                theme_tree_features[level_name][cat_id] = text_calculator.correct_domain_bias_iNat(
                    text_feature, domain_mean, cross_level_mean, domain_invariant_mean
                )


    # 遍历修正后的特征
    for level_name, level_ids in theme_tree_features.items():  # 修改点
        total_num = len(list(level_ids.values()))
        print(f"Total feats = {total_num} at {level_name}")

    # 准备并保存修正后的特征
    for level_name, level_theme_dict in theme_tree_features.items():  # 修改点
        sorted_theme_dict = OrderedDict(sorted(level_theme_dict.items(), key=lambda x: int(x[0])))

        l_feats = list(sorted_theme_dict.values())
        l_classifier = torch.stack(l_feats)
        print(f"---> {level_name}'s classifier has a shape of {l_classifier.shape}")

        # Save the embeddings
        path_save = os.path.join(args.out_path, f"fsod_clip_hrchy_{level_name}.npy")
        print(f'Saving to {path_save}')
        np.save(open(path_save, 'wb'), l_classifier.cpu().numpy())

    # for level_name, level_ids in theme_tree_features.items():
    #     total_num = len(list(level_ids.values()))
    #     print(f"Total feats = {total_num} at {level_name}")

    # # Prepare and Save Features
    # for level_name, level_theme_dict in theme_tree_features.items():
    #     sorted_theme_dict = OrderedDict(sorted(level_theme_dict.items(), key=lambda x: int(x[0])))

    #     l_feats = list(sorted_theme_dict.values())
    #     l_classifier = torch.stack(l_feats)
    #     print(f"---> {level_name}'s classifier has a shape of {l_classifier.shape}")
    #     # Save the embeddings
    #     path_save = os.path.join(args.out_path, f"fsod_clip_hrchy_{level_name}.npy")
    #     print(f'Saving to {path_save}')
    #     # torch.save(l_classifier.cpu(), path_save)
    #     np.save(open(path_save, 'wb'), l_classifier.cpu().numpy())





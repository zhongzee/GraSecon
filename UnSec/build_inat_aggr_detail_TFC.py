import argparse
import torch
import numpy as np
import clip
from collections import defaultdict, OrderedDict
from copy import deepcopy
from tools.composer import SignatureComposer
from tools.themer import Themer
from tools.fileios import *


def visualize_feature_comparison(theme_tree_before, theme_tree_after, level_names, save_dir, method='tsne', perplexity=30, n_components=2):
    """
    可视化修正前后特征的对比图，使用 PCA 或 t-SNE 进行降维。

    Args:
        theme_tree_before (dict): 修正前的 theme_tree_features。
        theme_tree_after (dict): 修正后的 theme_tree_features。
        level_names (list): 层级名称列表，如 ['l1', 'l2', 'l3']。
        save_dir (str): 保存图像的目录。
        method (str): 降维方法，'pca' 或 'tsne'。
        perplexity (int): t-SNE 的 perplexity 参数（仅当 method='tsne' 时使用）。
        n_components (int): 降维后的维度数，默认 2。
    """
    os.makedirs(save_dir, exist_ok=True)

    for level in level_names:
        features_before = theme_tree_before.get(level, {})
        features_after = theme_tree_after.get(level, {})
        
        if not features_before or not features_after:
            print(f"No features found for level {level}. Skipping.")
            continue
        
        # 收集所有类别 ID
        class_ids = list(set(features_before.keys()).union(features_after.keys()))
        class_ids = sorted(class_ids)  # 保持顺序一致
        num_classes = len(class_ids)
        
        if num_classes == 0:
            print(f"No classes found for level {level}. Skipping.")
            continue
        
        # 为每个类别分配一个唯一颜色
        if num_classes <= 20:
            cmap = cm.get_cmap('tab20', num_classes)
        elif num_classes <= 40:
            cmap = cm.get_cmap('tab40', num_classes)
        elif num_classes <= 256:
            cmap = cm.get_cmap('hsv', num_classes)
        else:
            cmap = cm.get_cmap('hsv', num_classes)
        colors = [cmap(i) for i in range(num_classes)]
        color_dict = {cid: colors[i % num_classes] for i, cid in enumerate(class_ids)}
        
        # 准备修正前的数据
        data_before = []
        labels_before = []
        for cid in class_ids:
            feats = features_before.get(cid, [])
            if isinstance(feats, list):
                for feat in feats:
                    data_before.append(feat.cpu().numpy())
                    labels_before.append(cid)
            else:
                data_before.append(feats.cpu().numpy())
                labels_before.append(cid)
        
        # 准备修正后的数据
        data_after = []
        labels_after = []
        for cid in class_ids:
            feat = features_after.get(cid)
            if feat is not None:
                data_after.append(feat.cpu().numpy())
                labels_after.append(cid)
        
        # 转换为 NumPy 数组
        data_before = np.array(data_before)
        labels_before = np.array(labels_before)
        data_after = np.array(data_after)
        labels_after = np.array(labels_after)
        
        if len(data_before) == 0 and len(data_after) == 0:
            print(f"No data to plot for level {level}. Skipping.")
            continue
        
        # 合并数据
        if len(data_before) > 0 and len(data_after) > 0:
            combined_data = np.vstack((data_before, data_after))
            combined_labels = np.concatenate((labels_before, labels_after))
            combined_markers = ['before'] * len(data_before) + ['after'] * len(data_after)
        elif len(data_before) > 0:
            combined_data = data_before
            combined_labels = labels_before
            combined_markers = ['before'] * len(data_before)
        else:
            combined_data = data_after
            combined_labels = labels_after
            combined_markers = ['after'] * len(data_after)
        
        # 应用 PCA 或 t-SNE
        if method == 'pca':
            reducer = PCA(n_components=n_components)
            reduced_data = reducer.fit_transform(combined_data)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, init='random')
            reduced_data = reducer.fit_transform(combined_data)
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")
        
        # 分割修正前后的降维数据
        if len(data_before) > 0 and len(data_after) > 0:
            reduced_before = reduced_data[:len(data_before)]
            reduced_after = reduced_data[len(data_before):]
            labels_before = labels_before
            labels_after = labels_after
        elif len(data_before) > 0:
            reduced_before = reduced_data
            labels_before = labels_before
            reduced_after = np.array([])
            labels_after = np.array([])
        else:
            reduced_before = np.array([])
            labels_before = np.array([])
            reduced_after = reduced_data
            labels_after = labels_after
        
        # 开始绘图
        plt.figure(figsize=(16, 8))
        
        # 绘制修正前的特征
        if len(reduced_before) > 0:
            for cid in class_ids:
                idxs = labels_before == cid
                if np.sum(idxs) > 0:
                    plt.scatter(reduced_before[idxs, 0], reduced_before[idxs, 1], 
                                c=[color_dict[cid]], marker='o', label=f'{cid} before' if cid == class_ids[0] else "", 
                                alpha=0.5, edgecolors='none')
        
        # 绘制修正后的特征
        if len(reduced_after) > 0:
            for cid in class_ids:
                idxs = labels_after == cid
                if np.sum(idxs) > 0:
                    plt.scatter(reduced_after[idxs, 0], reduced_after[idxs, 1], 
                                c=[color_dict[cid]], marker='x', label=f'{cid} after' if cid == class_ids[0] else "", 
                                alpha=0.5, edgecolors='none')
        
        plt.title(f'Feature Visualization for Level {level}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        # 为避免图例重复，只保留一次标签
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', markerscale=1)
        
        plt.grid(True)
        
        # 保存图像
        save_path = os.path.join(save_dir, f'feature_visualization_{level}.png')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization for level {level} to {save_path}")

def load_detail_sentences(hierarchy_dir, level_names):
    """
    Load detail_sentences for each level from JSON files into a dictionary.
    """
    detail_dict = {}
    for level in level_names:
        file_path = os.path.join(hierarchy_dir, f"cleaned_inat_gpt_detail_hrchy_{level}.json")
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
    l_names = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']

    meta_level_leaf = htree.get(leaf_level)

    # Extract class ids with their parents
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
            this_name = htree[level_name]['categories'][this_id - 1]['name']
            cat_name.append(this_name)
        signature_names.append(cat_name)

    assert len(signature_ids) == len(signature_names)
    assert all(len(signature_id) == len(signature_name) for signature_id, signature_name in
               zip(signature_ids, signature_names))

    # Compose sentences from signatures
    sentences = composer.compose(signature_names)

    for sent in sentences:
        print("orin_sent",sent)

    # Match and append detail sentences to each signature
    detail_sentences = []
    for signature in signature_names:
        details = []
        for name in signature:
            if name in detail_dict:
                details.extend(detail_dict[name])
        detail_sentences.append(details)

    for sent in detail_sentences:
        print("detai_sent",sent)

    print('Loading CLIP')
    model, preprocess = clip.load(clip_model, device=device)

    # Encode sentences
    text = clip.tokenize(sentences).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)

    # Encode detail sentences and combine
    detail_features = []

    # Encode detail_sentences and calculate mean
    for detail_set in detail_sentences:
        if not detail_set:
            continue
        detail_tokens = clip.tokenize(detail_set).to(device)
        with torch.no_grad():
            encoded_details = model.encode_text(detail_tokens)
        avg_detail_feature = get_mean_vector(encoded_details)  # Use _get_mean_vector
        detail_features.append(avg_detail_feature)

    # Ensure detail features match text features length
    assert len(detail_features) == len(text_features), "Mismatch between sentences and detail_sentences encoding."

    # Combine features
    combined_features = []
    for text_feat, detail_feat in zip(text_features, detail_features):
        stacked_feats = torch.stack([text_feat, detail_feat])
        combined = get_mean_vector(stacked_feats)  # Use _get_mean_vector
        combined_features.append(combined)

    # Return combined features
    return torch.stack(combined_features)


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', default='./UnSec/inat_annotations/inat_hierarchy_tree.json')
    parser.add_argument('--prompter', default='isa', choices=['a', 'concat', 'isa'])
    parser.add_argument('--aggregator', default='mean', choices=['peigen', 'mean', 'mixed', 'all_eigens'])
    parser.add_argument('--out_path', default='./nexus/inat/vitB32/UnSec_inat_detail_TFC')
    parser.add_argument('--dataset_name', default='inat', choices=['inat', 'fsod'])
    parser.add_argument('--peigen_thresh', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--clip_model', default="ViT-B/32", choices=['ViT-B/32', 'RN50'])
    parser.add_argument('--enable_global_mean', action='store_true', default=False, help='是否开启多领域总体均值校正') # iNAT不需要

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
    level_names = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']

    meta_tree = json.load(open(args.tree, 'r'))
    meta_level_leaf = meta_tree.get('l6')

    # Load detail sentences
    hierarchy_dir = "./UnSec/inat_llm_detail_answers/"
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
        'l3': {},
        'l4': {},
        'l5': {},
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
        if level_name == 'l6':
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
    corrected_tree_features = apply_granularity_bias_correction(theme_tree_features, mean_features, global_mean)


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
        path_save = os.path.join(args.out_path, f"inat_clip_hrchy_{level_name}.npy")
        print(f'Saving to {path_save}')
        np.save(open(path_save, 'wb'), l_classifier.cpu().numpy())

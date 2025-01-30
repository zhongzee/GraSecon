import argparse
import torch
import numpy as np
import clip
from collections import defaultdict, OrderedDict
from copy import deepcopy
import json
import os
from tools.composer import SignatureComposer
from tools.themer import Themer
from tools.fileios import *  # 确保这些模块中包含 is_valid_folder 函数
from tools.IFC2 import TextMeanFeatureCalculator2
import logging


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm


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

def compute_leaf_embedding(htree, leaf_level, composer, clip_model='ViT-B/32', device='cuda'):
    l_names = ['l6', 'l5', 'l4','l3', 'l2', 'l1']
    meta_level_leaf = htree.get(leaf_level)

    # extract class ids w/ its parents
    signature_ids = [[int(x['id'])] for x in sorted(meta_level_leaf['categories'], key=lambda x: x['id'])]
    for i in range(len(signature_ids)):
        leaf_id = str(signature_ids[i][0])
        parents_ids = meta_level_leaf['parents'].get(leaf_id, [])
        signature_ids[i].extend(parents_ids)

    signature_names = []
    for cat_id in signature_ids:
        cat_name = []
        for level_idx, this_id in enumerate(cat_id):
            level_name = l_names[level_idx]
            try:
                this_name = htree[level_name]['categories'][this_id-1]['name']
            except IndexError:
                this_name = "Unknown"
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
    return text_features  # F.normalize(text_features)


# 新增函数：验证父子节点映射
def validate_parent_child_mappings(level_hierarchy, meta_tree, category_id_to_name):
    """
    验证所有 parent_ids 和 child_ids 是否在其对应层级中都有映射。

    Args:
        level_hierarchy (list): 层级名称按从高到低排序的列表，如 ['l1', 'l2', 'l3']。
        meta_tree (dict): 层级树的完整字典结构。
        category_id_to_name (defaultdict): {level_name: {unique_id: category_name}}

    Returns:
        missing_parents (set): 缺少映射的父节点集合。
        missing_children (set): 缺少映射的子节点集合。
    """
    missing_parents = set()
    missing_children = set()

    for level_idx, level_name in enumerate(level_hierarchy):
        current_level = meta_tree.get(level_name, {})
        categories = current_level.get('categories', [])
        parents = current_level.get('parents', {})
        childs = current_level.get('childs', {})

        # 验证父节点
        for cat in categories:
            cat_id = str(cat['id'])
            parent_ids = parents.get(cat_id, [])
            if parent_ids:
                if level_idx + 1 >= len(level_hierarchy):
                    # 没有更高层级存在
                    missing_parents.add(f"Level {level_name}: No higher level for parents {parent_ids} of category ID {cat_id}")
                    continue
                parent_level = level_hierarchy[level_idx + 1]
                for parent_id in parent_ids:
                    parent_unique_id = f"{parent_level}_{parent_id}"
                    if parent_unique_id not in category_id_to_name[parent_level]:
                        missing_parents.add(f"Level {level_name}: Parent ID {parent_id} not found in level {parent_level}")

        # 验证子节点
        for cat in categories:
            cat_id = str(cat['id'])
            child_ids = childs.get(cat_id, [])
            if child_ids:
                if level_idx - 1 < 0:
                    # 没有更低层级存在
                    missing_children.add(f"Level {level_name}: No lower level for children {child_ids} of category ID {cat_id}")
                    continue
                child_level = level_hierarchy[level_idx - 1]
                for child_id in child_ids:
                    child_unique_id = f"{child_level}_{child_id}"
                    if child_unique_id not in category_id_to_name[child_level]:
                        missing_children.add(f"Level {level_name}: Child ID {child_id} not found in level {child_level}")

    return missing_parents, missing_children


# 新增函数：计算均值特征
import os
import numpy as np
import torch
import logging

def compute_mean_features(save_dir, dataset_name, theme_tree_features, device, sentence_type="detail"):
    """
    计算每个层级的均值特征，若已存在则加载，否则计算并保存。

    Args:
        save_dir (str): 保存目录。
        dataset_name (str): 数据集名称，用于区分保存路径。
        theme_tree_features (dict): 主题树特征，键为层级名称，值为特征字典。
        device (str): 设备类型，如 'cuda' 或 'cpu'。
        sentence_type (str): 使用的句子类型，可选 "by_level", "candidate", "detail", "combined"。
        max_length (int): 对特征进行统一的最大长度（仅在特定场景下生效）。

    Returns:
        mean_features (dict): 每个层级的均值特征，键为层级名称，值为均值特征张量。
    """
    logger = logging.getLogger("TextMeanFeatureCalculator")
    if not logger.handlers:
        # 配置日志处理器（如果尚未配置）
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    # 构建保存目录
    dataset_save_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_save_dir, exist_ok=True)
    
    # 构建每个层级的均值特征文件路径
    mean_feature_paths = {level: os.path.join(dataset_save_dir, f"{level}_mean_features.npy") 
                          for level in theme_tree_features.keys()}
    mean_features = {}

    # 检查是否已有保存的均值特征
    all_exists = all(os.path.exists(path) for path in mean_feature_paths.values())
    if all_exists:
        for level, path in mean_feature_paths.items():
            mean_features[level] = torch.from_numpy(np.load(path)).to(device)
        logger.info(f"已加载数据集 {dataset_name} 的所有层级均值特征")
        return mean_features

    # 遍历每个层级，计算或加载均值特征
    for level, features_dict in theme_tree_features.items():
        if len(features_dict) == 0:
            logger.warning(f"没有找到层级 '{level}' 的特征，跳过均值计算")
            continue

        # Flatten the list of features: if unique_id maps to list, extend; else, append
        features = []
        for unique_id, feat in features_dict.items():
            if isinstance(feat, list):
                features.extend(feat)
            else:
                features.append(feat)

        if len(features) == 0:
            logger.warning(f"层级 '{level}' 的特征列表为空，跳过均值计算")
            continue

        # 根据 sentence_type 选择不同的均值计算策略
        if sentence_type in ["combined", "by_level", "candidate"]:
            logger.info(f"层级 '{level}' 使用逐步累积方式计算均值特征")
            # 检查每个特征的维度是否一致
            feature_dim = features[0].shape[-1]
            for feature in features:
                if feature.shape[-1] != feature_dim:
                    raise ValueError(
                        f"层级 '{level}' 中的特征维度不一致：期望 {feature_dim}, 但得到 {feature.shape[-1]}"
                    )
            
            # 使用逐步计算均值的方式
            sum_feature = torch.zeros_like(features[0])  # 初始化累积张量
            count = 0
            for feature in features:
                # 逐步累积特征的均值
                sum_feature += feature.mean(dim=0)
                count += 1
            mean_feature = sum_feature / count  # 计算均值
        else:
            logger.info(f"层级 '{level}' 使用堆叠方式计算均值特征")
            # 确保每个 feature 是 [C] 的形状
            features = [f.squeeze(0) if f.dim() == 2 and f.size(0) == 1 else f for f in features]

            # 堆叠特征
            stacked_features = torch.stack(features)  # [N, C]

            # 检查堆叠后的维度是否正确
            if stacked_features.dim() != 2:
                raise ValueError(f"层级 '{level}' 的堆叠特征维度不正确：{stacked_features.shape}")
            
            # 计算均值特征
            mean_feature = stacked_features.mean(dim=0)  # [C]

        # 保存均值特征
        mean_features[level] = mean_feature
        np.save(mean_feature_paths[level], mean_feature.cpu().numpy())  # 保存到文件
        logger.info(f"已保存层级 '{level}' 的均值特征到 {mean_feature_paths[level]}")

    return mean_features


# 新增函数：校正领域偏置
def correct_domain_bias_fsod(text_features, domain_mean, cross_level_mean=None, global_mean=None, parent_mean=None, child_mean=None, level_name=None, delta=0.1):
    """
    校正文本特征的领域偏置。

    Args:
        text_features (torch.Tensor): 文本特征，形状为 [C] 或 [N, C]。
        domain_mean (torch.Tensor): 当前层级的均值特征，形状为 [C]。
        cross_level_mean (torch.Tensor, optional): 跨层级的均值特征，形状为 [C]。
        global_mean (torch.Tensor, optional): 全局均值特征，形状为 [C]。
        parent_mean (torch.Tensor, optional): 父节点的均值特征，形状为 [C]。
        child_mean (torch.Tensor, optional): 子节点的均值特征，形状为 [C]。
        level_name (str, optional): 当前层级名称，用于动态加权策略。
        delta (float): 偏置向量幅度限制。

    Returns:
        torch.Tensor: 校正后的文本特征，形状与 text_features 相同。
    """
    # 检查输入的形状是否符合预期
    if text_features.dim() == 1:  # 如果是 [C]
        text_features = text_features.unsqueeze(0)  # 转换为 [1, C]
    elif text_features.dim() != 2:
        raise ValueError(f"text_features shape is invalid: {text_features.shape}, expected [C] or [N, C]")

    if domain_mean.dim() != 1:
        raise ValueError(f"domain_mean shape is invalid: {domain_mean.shape}, expected [C]")

    # 获取动态权重
    if level_name is not None:
        alpha, beta, gamma = get_dynamic_weights(level_name)
    else:
        alpha, beta, gamma = 1.0, 0.5, 0.2  # 默认权重

    # 初始化偏置向量 t_hat
    t_hat = alpha * domain_mean.clone()

    # 动态加权策略
    if cross_level_mean is not None:
        t_hat += beta * cross_level_mean

    if global_mean is not None:
        t_hat += gamma * global_mean

    # 引入父子关系约束
    if parent_mean is not None:
        t_hat = domain_mean - parent_mean
        t_hat = torch.clamp(t_hat, min=parent_mean - delta, max=parent_mean + delta)

    # 子节点增强语义独立性
    if child_mean is not None:
        t_hat += 0.2 * (t_hat - child_mean)

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
        return corrected_text_features.squeeze(0)

    return corrected_text_features


# 获取动态权重的函数
def get_dynamic_weights(level_name):
    """
    根据层级名称获取动态权重因子 (alpha, beta, gamma)。

    Args:
        level_name (str): 层级名称，如 'l1', 'l2', 'l3'。

    Returns:
        tuple: (alpha, beta, gamma) 权重因子。
    """
    # 这里可以根据不同层级名称定义不同的权重策略
    # 例如，对于细粒度层级，可能需要较低的 alpha，较高的 beta 和 gamma
    weights = {
        'l1': (0.3, 0.1, 0.05),
        'l2': (0.5, 0.2, 0.1),
        'l3': (1.0, 0.5, 0.2)
    }
    return weights.get(level_name, (1.0, 0.5, 0.2))  # 默认权重


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', default='./UnSec/inat_annotations/inat_hierarchy_tree.json')
    parser.add_argument('--prompter', default='isa', choices=['a', 'concat', 'isa'])
    parser.add_argument('--aggregator', default='mean', choices=['peigen', 'mean', 'mixed', 'all_eigens'])
    parser.add_argument('--out_path', default='./nexus/inat/vitB32/UnSec_GT_by_level_IFC')

    parser.add_argument('--peigen_thresh', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--clip_model', default="ViT-B/32", choices=['ViT-B/32', 'RN50'])
    parser.add_argument('--dataset_name', default='inat', choices=['inat', 'fsod'])
    parser.add_argument('--sentence_type', default='by_level', choices=['by_level', 'candidate', 'detail', 'combined'])
    parser.add_argument('--enable_global_mean', action='store_true', default=False, help='是否开启多领域总体均值校正')
    parser.add_argument('--enable_cross_level_mean', action='store_true', default=True, help='是否开启跨粒度层级的均值校正')

    args = parser.parse_args()
    logger = logging.getLogger("Correct_domain")

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    if not is_valid_folder(args.out_path):
        raise FileExistsError(f"Invalid output folder: {args.out_path}")

    # Device Selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print('Loading CLIP')
    global_encoder, global_preprocess = clip.load(args.clip_model, device=device)

    # Load Metadata
    print(f'Loading {args.tree}')
    if args.dataset_name == "fsod":
        level_names = ['l3', 'l2', 'l1']  # 根据数据集的层级结构调整顺序
    else:
        level_names = ['l6', 'l5', 'l4','l3', 'l2', 'l1']

    meta_tree = json.load(open(args.tree, 'r'))
    meta_level_leaf = meta_tree.get('l3')  # 假设 'l3' 是叶子层级，根据实际情况调整

    # extract class ids w/ its parents
    signature_ids = [[int(x['id'])] for x in sorted(meta_level_leaf['categories'], key=lambda x: x['id'])]
    for i in range(len(signature_ids)):
        leaf_id = str(signature_ids[i][0])
        parents_ids = meta_level_leaf['parents'].get(leaf_id, [])
        signature_ids[i].extend(parents_ids)

    if args.dataset_name == "fsod":
        tree_childs_to_leaf = {
            'l1': {},
            'l2': {},
        }
    else:
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
            if level_idx >= len(leaf_signature):
                break  # 防止索引越界
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
    
    text_calculator = TextMeanFeatureCalculator2(save_dir='./nexus/inat/vitB32/UnSec_GT_by_level_IFC', device=device)

    print(leaf_features.shape)

    tree_features = defaultdict(dict)
    for i, feat in enumerate(leaf_features, start=1):
        tree_features[level_names[0]][str(i)] = feat

    for level_name, level_childs in tree_childs_to_leaf.items():
        for level_cat_id, level_child_ids in level_childs.items():
            print(f"{level_name} cat_{level_cat_id} has childs: {level_child_ids}")

            level_child_feats = []
            for idx in level_child_ids:
                child_feat = tree_features[level_names[0]].get(str(idx), None)
                if child_feat is not None:
                    level_child_feats.append(child_feat)
                else:
                    print(f"Warning: Child feature '{str(idx)}' not found in level '{level_names[0]}'")

            if not level_child_feats:
                print(f"Warning: No valid child features found for {level_name}_{level_cat_id}")
                continue

            tree_features[level_name][level_cat_id] = level_child_feats

    theme_tree_features = deepcopy(tree_features)

    # 新增调用：验证父子节点映射
    # -----------------------------------
    # 这个步骤将验证所有的 parent_ids 和 child_ids 是否在对应的层级中有映射
    category_id_to_name = defaultdict(dict, {
        lvl: {f"{lvl}_{cat['id']}": cat['name'].lower() for cat in meta_tree[lvl]['categories']}
        for lvl in level_names
    })

    missing_parents, missing_children = validate_parent_child_mappings(
        level_hierarchy=level_names,
        meta_tree=meta_tree,
        category_id_to_name=category_id_to_name
    )

    if missing_parents:
        print(f"Missing parent mappings for: {missing_parents}")
    else:
        print("All parent_ids have corresponding categories.")

    if missing_children:
        print(f"Missing child mappings for: {missing_children}")
    else:
        print("All child_ids have corresponding categories.")
    # -----------------------------------

    # 新增调用：计算均值特征
    # -----------------------------------
    # 这个步骤将计算每个层级的均值特征
    
    theme_tree_before_correction = deepcopy(theme_tree_features)

    # mean_features = text_calculator.compute_mean_features(args.dataset_name, all_text_features, sentence_type=args.sentence_type)
    # 计算或加载均值特征
    mean_features = compute_mean_features(
        save_dir=args.out_path,
        dataset_name=args.dataset_name,
        theme_tree_features=theme_tree_features,
        device=device,
        sentence_type=args.sentence_type
    )
    # -----------------------------------

    # 新增调用：应用领域偏置修正
    # -----------------------------------
    # 这个步骤将应用领域偏置修正到主题树特征
    # ...（前面的代码保持不变）

    # 继续处理主题特征（如聚合等）
    for level_name, level_ids in tree_features.items():
        if args.dataset_name =="fsod" and level_name == 'l3':
            continue  # 假设 'l3' 是叶子层级，无需进一步聚合
        if args.dataset_name =="inat" and level_name == 'l6':
            continue  # 假设 'l3' 是叶子层级，无需进一步聚合

        for level_cat_id, level_child_feats in level_ids.items():
            if not isinstance(level_child_feats, list):
                logger.warning(f"Expected a list of child features for {level_name}_{level_cat_id}, but got {type(level_child_feats)}")
                continue

            if len(level_child_feats) == 0:
                logger.warning(f"No child features found for {level_name}_{level_cat_id}")
                continue

            # 堆叠子节点特征
            try:
                stacked_child_feats = torch.stack(level_child_feats)
            except Exception as e:
                logger.error(f"Error stacking child features for {level_name}_{level_cat_id}: {e}")
                continue

            # 聚合子节点特征，假设使用均值
            theme_feat = stacked_child_feats.mean(dim=0)  # [C]
            # 或者使用其他聚合方法，如 PCA 等，根据您的需求调整

            # 将聚合后的特征赋值回 theme_tree_features
            theme_tree_features[level_name][level_cat_id] = theme_feat

    # 现在，theme_tree_features[level][id] 是单一的 Tensor
    # 进行偏置修正
    for level_name, level_data in theme_tree_features.items():
        for unique_id, text_feature in level_data.items():
            if level_name not in mean_features:
                logger.warning(f"Mean feature for level '{level_name}' not found. Skipping bias correction.")
                continue

            domain_mean = mean_features[level_name]

            # 计算跨层级均值
            cross_level_mean = None
            if args.enable_cross_level_mean:
                cross_level_mean = torch.stack(list(mean_features.values())).mean(dim=0)

            # 计算全局均值
            global_mean_features = None
            if args.enable_global_mean:
                global_mean_features = torch.stack(list(mean_features.values())).mean(dim=0)

            # 获取当前节点的父节点和子节点信息
            current_level = level_name

            # 根据层级名称解析 unique_id
            if '_' in unique_id:
                # 例如 'cat_28'
                split_id = unique_id.split('_')
                if len(split_id) > 1:
                    current_cat_id = split_id[1]
                else:
                    current_cat_id = split_id[0]
                    logger.warning(f"Unique_id '{unique_id}' in level '{level_name}' does not contain '_'. Using as is.")
            else:
                # 例如 '1' 在 'l3' 中
                current_cat_id = unique_id

            parent_ids = meta_tree[current_level]['parents'].get(current_cat_id, [])
            child_ids = meta_tree[current_level]['childs'].get(current_cat_id, [])

            # 计算父节点均值
            parent_mean = None
            if parent_ids:
                # 将 parent_ids 转换为字符串，以匹配 theme_tree_features 的键
                parent_unique_ids = [str(pid) for pid in parent_ids if str(pid) in theme_tree_features[level_names[level_names.index(level_name)+1]]]
                parent_features = [theme_tree_features[level_names[level_names.index(level_name)+1]].get(pid, None) for pid in parent_unique_ids]
                parent_features = [f for f in parent_features if f is not None]
                if parent_features:
                    parent_mean = torch.stack(parent_features).mean(dim=0)
                else:
                    logger.warning(f"No valid parent features found for unique_id '{unique_id}' in level '{level_name}'")

            # 计算子节点均值（此时子节点特征已经被聚合为单一的 Tensor）
            child_mean = None
            if child_ids:
                lower_level_idx = level_names.index(level_name) - 1
                if lower_level_idx >= 0:
                    lower_level = level_names[lower_level_idx]
                    # 判断下一级 unique_id 是否带 'cat_' 前缀
                    # if lower_level == 'l2':
                    #     child_unique_ids = [f"cat_{cid}" for cid in child_ids if f"cat_{cid}" in theme_tree_features[lower_level]]
                    # else:
                    child_unique_ids = [str(cid) for cid in child_ids if str(cid) in theme_tree_features[lower_level]]

                    child_features = [theme_tree_features[lower_level].get(cid, None) for cid in child_unique_ids]
                    child_features = [f for f in child_features if f is not None]
                    if child_features:
                        child_mean = torch.stack(child_features).mean(dim=0)
                    else:
                        logger.warning(f"No valid child features found for unique_id '{unique_id}' in level '{level_name}'")

            # 应用偏置修正
            corrected_feature = correct_domain_bias_fsod(
                text_features=text_feature,
                domain_mean=domain_mean,
                cross_level_mean=cross_level_mean,
                global_mean=global_mean_features,
                parent_mean=parent_mean,
                child_mean=child_mean,
                level_name=level_name,
                delta=0.1
            )

            theme_tree_features[level_name][unique_id] = corrected_feature
            node_name = category_id_to_name[level_name].get(unique_id, 'Unknown')
            logger.info(f"Applied domain bias correction for '{unique_id}' ('{node_name}') at level '{level_name}'")

    # 在应用偏置修正之后，保存一份拷贝
    theme_tree_after_correction = deepcopy(theme_tree_features)

     # 调用可视化函数，生成对比图
    visualize_feature_comparison(
        theme_tree_before=theme_tree_before_correction,
        theme_tree_after=theme_tree_after_correction,
        level_names=level_names,
        save_dir=args.out_path,
        method='tsne',  # 可选择 'pca' 或 'tsne'
        perplexity=30,   # t-SNE 参数，仅在 method='tsne' 时使用
        n_components=2    # 降维到 2 维
    )

    # 根据需要，继续处理主题特征
    for level_name, level_ids in tree_features.items():
        if level_name == 'l3':
            continue

        for level_cat_id, level_child_feats in level_ids.items():
            stacked_child_feats = torch.stack(level_child_feats)
            theme_feat = theme_maker.get_theme(stacked_child_feats)
            theme_tree_features[level_name][level_cat_id] = theme_feat

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
        path_save = os.path.join(args.out_path, f"fsod_clip_hrchy_{level_name}.npy")
        print(f'Saving to {path_save}')
        # torch.save(l_classifier.cpu(), path_save)
        np.save(open(path_save, 'wb'), l_classifier.cpu().numpy())


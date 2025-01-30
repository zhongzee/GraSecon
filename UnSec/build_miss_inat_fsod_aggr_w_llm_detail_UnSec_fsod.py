import argparse
import torch
import numpy as np
import clip
from collections import defaultdict, OrderedDict
from copy import deepcopy
from tools.themer import Themer
from tools.fileios import *
from torch.nn import functional as F
# from sentence_transformers import SentenceTransformer, util

coco_novel = [
    'airplane',
    'bus',
    'cat',
    'dog',
    'cow',
    'elephant',
    'umbrella',
    'tie',
    'snowboard',
    'skateboard',
    'cup',
    'knife',
    'cake',
    'couch',
    'keyboard',
    'sink',
    'scissors',
]

lvis_novel = ['applesauce', 'apricot', 'arctic (type of shoe)', 'armoire', 'armor', 'ax', 'baboon', 'bagpipe', 'baguet', 'bait', 'ballet skirt', 'banjo', 'barbell', 'barge', 'bass horn', 'batter (food)', 'beachball', 'bedpan', 'beeper', 'beetle', 'bible', 'birthday card', 'pirate flag', 'blimp', 'gameboard', 'bob', 'bolo tie', 'bonnet', 'bookmark', 'boom microphone', 'bow (weapon)', 'pipe bowl', 'bowling ball', 'boxing glove', 'brass plaque', 'breechcloth', 'broach', 'bubble gum', 'horse buggy', 'bulldozer', 'bulletproof vest', 'burrito', 'cabana', 'locker', 'candy bar', 'canteen', 'elevator car', 'car battery', 'cargo ship', 'carnation', 'casserole', 'cassette', 'chain mail', 'chaise longue', 'chalice', 'chap', 'checkbook', 'checkerboard', 'chessboard', 'chime', 'chinaware', 'poker chip', 'chocolate milk', 'chocolate mousse', 'cider', 'cigar box', 'clarinet', 'cleat (for securing rope)', 'clementine', 'clippers (for plants)', 'cloak', 'clutch bag', 'cockroach', 'cocoa (beverage)', 'coil', 'coloring material', 'combination lock', 'comic book', 'compass', 'convertible (automobile)', 'sofa bed', 'cooker', 'cooking utensil', 'corkboard', 'cornbread', 'cornmeal', 'cougar', 'coverall', 'crabmeat', 'crape', 'cream pitcher', 'crouton', 'crowbar', 'hair curler', 'curling iron', 'cylinder', 'cymbal', 'dagger', 'dalmatian', 'date (fruit)', 'detergent', 'diary', 'die', 'dinghy', 'tux', 'dishwasher detergent', 'diving board', 'dollar', 'dollhouse', 'dove', 'dragonfly', 'drone', 'dropper', 'drumstick', 'dumbbell', 'dustpan', 'earplug', 'eclair', 'eel', 'egg roll', 'electric chair', 'escargot', 'eyepatch', 'falcon', 'fedora', 'ferret', 'fig (fruit)', 'file (tool)', 'first aid kit', 'fishbowl', 'flash', 'fleece', 'football helmet', 'fudge', 'funnel', 'futon', 'gag', 'garbage', 'gargoyle', 'gasmask', 'gemstone', 'generator', 'goldfish', 'gondola (boat)', 'gorilla', 'gourd', 'gravy boat', 'griddle', 'grits', 'halter top', 'hamper', 'hand glass', 'handcuff', 'handsaw', 'hardback book', 'harmonium', 'hatbox', 'headset', 'heron', 'hippopotamus', 'hockey stick', 'hookah', 'hornet', 'hot air balloon', 'hotplate', 'hourglass', 'houseboat', 'hummus', 'popsicle', 'ice pack', 'ice skate', 'inhaler', 'jelly bean', 'jewel', 'joystick', 'keg', 'kennel', 'keycard', 'kitchen table', 'knitting needle', 'knocker (on a door)', 'koala', 'lab coat', 'lamb chop', 'lasagna', 'lawn mower', 'leather', 'legume', 'lemonade', 'lightning rod', 'limousine', 'liquor', 'machine gun', 'mallard', 'mallet', 'mammoth', 'manatee', 'martini', 'mascot', 'masher', 'matchbox', 'microscope', 'milestone', 'milk can', 'milkshake', 'mint candy', 'motor vehicle', 'music stool', 'nailfile', 'neckerchief', 'nosebag (for animals)', 'nutcracker', 'octopus (food)', 'octopus (animal)', 'omelet', 'inkpad', 'pan (metal container)', 'pantyhose', 'papaya', 'paperback book', 'paperweight', 'parchment', 'passenger ship', 'patty (food)', 'wooden leg', 'pegboard', 'pencil box', 'pencil sharpener', 'pendulum', 'pennant', 'penny (coin)', 'persimmon', 'phonebook', 'piggy bank', 'pin (non jewelry)', 'ping pong ball', 'pinwheel', 'tobacco pipe', 'pistol', 'pitchfork', 'playpen', 'plow (farm equipment)', 'plume', 'pocket watch', 'poncho', 'pool table', 'prune', 'pudding', 'puffer (fish)', 'puffin', 'pug dog', 'puncher', 'puppet', 'quesadilla', 'quiche', 'race car', 'radar', 'rag doll', 'rat', 'rib (food)', 'river boat', 'road map', 'rodent', 'roller skate', 'rollerblade', 'root beer', 'safety pin', 'salad plate', 'salmon (food)', 'satchel', 'saucepan', 'sawhorse', 'saxophone', 'scarecrow', 'scraper', 'seaplane', 'sharpener', 'sharpie', 'shaver (electric)', 'shawl', 'shears', 'shepherd dog', 'sherbert', 'shot glass', 'shower cap', 'shredder (for paper)', 'skullcap', 'sling (bandage)', 'smoothie', 'snake', 'softball', 'sombrero', 'soup bowl', 'soya milk', 'space shuttle', 'sparkler (fireworks)', 'spear', 'crawfish', 'squid (food)', 'stagecoach', 'steak knife', 'stepladder', 'stew', 'stirrer', 'string cheese', 'stylus', 'subwoofer', 'sugar bowl', 'sugarcane (plant)', 'syringe', 'tabasco sauce', 'table tennis table', 'tachometer', 'taco', 'tambourine', 'army tank', 'telephoto lens', 'tequila', 'thimble', 'trampoline', 'trench coat', 'triangle (musical instrument)', 'truffle (chocolate)', 'vat', 'turnip', 'unicycle', 'vinegar', 'violin', 'vodka', 'vulture', 'waffle iron', 'walrus', 'wardrobe', 'washbasin', 'water heater', 'water gun', 'wolf']

"""
python build_miss_inat_fsod_aggr_w_llm_hrchy.py \
--dataset_name "lvis"
--gpt_results_root lvis_llm_answers
--prompter "isa"
--aggregator "mean"
--clip_model "ViT-B/32"
--out_path ".././nexus/lvis/UnSec_llm_detail_combine"

"""

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
    if sentence_type == "hierarchy":
        return candidate_sentences
    elif sentence_type == "detail":
        return detail_sentences
    elif sentence_type == "combined":
        return candidate_sentences + detail_sentences
    elif sentence_type == "by_level":
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='fsod_expanded', choices=['inat_expanded',
                                                                                         'fsod_expanded',
                                                                                         'coco', 'lvis', 'oid'])
    parser.add_argument('--gpt_results_root', default='./UnSec/fsod_llm_detail_answers')
    parser.add_argument('--prompter', default='isa', choices=['a', 'avg', 'concat', 'isa'])
    parser.add_argument('--aggregator', default='mean', choices=['peigen', 'mean', 'mixed', 'all_eigens',
                                                                              'plain'])
    parser.add_argument('--peigen_thresh', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--clip_model', default="ViT-B/32")
    parser.add_argument('--out_path', default='./nexus/fsod_miss/vitB32/UnSec_llm_detail_unsec')#UnSec_llm_detail_unsec_epoch1,UnSec_llm_detail_unsec_wo_SR
    parser.add_argument('--sentence_type', default='by_level', choices=['by_level', 'candidate', 'detail', 'combined'])
    parser.add_argument('--enable_global_mean', action='store_true', default=True, help='是否开启多领域总体均值校正') # 使用原本的修正方式前5层不需要开启gm第6层需要,使用最新的一直需要开启
    parser.add_argument('--enable_cross_level_mean', action='store_true', default=True, help='是否开启跨粒度层级的均值校正')
    parser.add_argument('--num_epochs', type=int, default=1, help="Number of epochs for optimization")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for optimization")
    parser.add_argument('--k', type=parse_k, default='all', help="Number of most relevant features to select, or 'all' to select all features")
    parser.add_argument('--optimizer', type=str, default='orin', choices=['orin', 'adm'], help="Select optimizer: 'adam' for traditional gradient descent, 'adm' for ADM")
    
    # inat_layer_policy = {
    # 'l1': 'no_gm',
    # 'l2': 'no_gm',
    # 'l3': 'no_gm',
    # 'l4': 'no_gm',
    # 'l5': 'no_gm',
    # 'l6': 'gm'
    # }

    layer_policy = {
        'l1': 'no_gm',
        'l2': 'gm',
        'l3': 'gm'
    }

    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    if not is_valid_folder(args.out_path): raise FileExistsError

    # Device Selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset_name in ["inat_expanded", "fsod_expanded"]:
        if args.dataset_name == 'inat_expanded':
            args.dataset_name = 'inat'
            level_names = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']
            # level_names = ['l1']
        else:
            args.dataset_name = 'fsod'
            level_names = ['l3', 'l2', 'l1']

        print('Loading CLIP')
        global_encoder, global_preprocess = clip.load(args.clip_model, device=device)

        theme_maker = Themer(method=args.aggregator if args.aggregator != "plain" else "mean",
                             thresh=args.peigen_thresh,
                             alpha=args.alpha)

        theme_tree_features = defaultdict(dict)
        all_text_features = defaultdict(list)
        
        for level_name in level_names[:1]:
            gpt_results = load_json(os.path.join(args.gpt_results_root,
                                                 f"cleaned_{args.dataset_name}_gpt_detail_hrchy_{level_name}.json"))

            expanded_results = load_json(f"./UnSec/miss_lvis_oid_llm_answers/cleaned_oid_lvis_gpt_detail_hrchy_{level_name}.json")

            # Removing overlapping node_names from B
            overlapping_names = {entry['node_name'].replace("_", " ").replace("-", " ").lower()
                                 for entry in gpt_results.values()}

            for key, value in list(expanded_results.items()):  # using list to create a copy so we can modify the expanded results in-place
                if value['node_name'] in overlapping_names:
                    del expanded_results[key]

            # Merging dictionaries with updated keys for the expanded categories
            next_key = len(gpt_results) + 1
            merged_dict = deepcopy(gpt_results)  # start with a copy of the target categories

            for value in expanded_results.values():
                merged_dict[str(next_key)] = value
                next_key += 1

            for cat_id, entry in sorted(merged_dict.items(), key=lambda item: int(item[0])):
                nodename = entry["node_name"]
                print(f"process {nodename},level_name {level_name}:")
                candidate_sentences = entry["candidate_sentences"] if args.aggregator != "plain" else entry["node_name"]
                detail_sentences = entry["detail_sentences"] if args.aggregator != "plain" else entry["node_name"]
                sentences_to_use = select_sentences_by_level(candidate_sentences, detail_sentences, level_name,
                                                             sentence_type=args.sentence_type)
                truncated_sentences = [sentence[:77] for sentence in sentences_to_use]
                node_tokens = clip.tokenize(truncated_sentences).to(device) # CLIP只处理candidate_sentences
                with torch.no_grad():
                    node_features = global_encoder.encode_text(node_tokens)
                node_features = F.normalize(node_features) # torch.Size([10, 512])
                # if node_features.size(0) == 1:
                #     print(f"level={level_name} cat={cat_id}")
                #     print(node_sentences)
                #     sys.exit()

                current_feature = generate_features(global_encoder, truncated_sentences, device, aggregation='mean').clone().detach().to(torch.float32)
                node_theme = theme_maker.get_theme(node_features).clone().detach().to(torch.float32)
                node_name_feature = get_node_name_feature(nodename, global_encoder, device).clone().detach().to(torch.float32)  # 已经是 float32

                # 在优化前调用
                # count_nan_inf(current_feature, "current_feature")
                # count_nan_inf(node_theme, "node_theme")

                if args.optimizer == 'adm':
                    print("当前优化层级：",level_name)
                    print("当前优化节点：",nodename)
                    optimized_feature = optimize_feature_with_adm_improved(##使用optimize_feature_with_adm_improved2效果很差
                        current_feature=node_theme,
                        node_name_feature=node_name_feature,
                        node_name=nodename,
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
                else:
                    optimized_feature = node_theme
                # 收集特征用于均值计算
                all_text_features[level_name].append(optimized_feature)
                theme_tree_features[level_name][cat_id] = optimized_feature

                # node_theme = theme_maker.get_theme(node_features) # 这里是平均的方式 torch.Size([512])
                # theme_tree_features[level_name][cat_id] = node_theme

        mean_features = compute_mean_features(theme_tree_features, save_dir=args.out_path, dataset_name=args.dataset_name, device=device)
    
        global_mean_features = None

        # Step 2: Compute global mean (μ_avg)
        global_mean = compute_global_mean(theme_tree_features)
        
        # Step 3：动态低层次使用gm
        corrected_tree_features = apply_layer_specific_bias_correction(theme_tree_features, mean_features, global_mean, layer_policy=layer_policy)#这个更好 38 42 66.6


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
            path_save = os.path.join(args.out_path, f"{args.dataset_name}_clip_hrchy_{level_name}.npy")

            print(f'Saving to {path_save}')
            np.save(open(path_save, 'wb'), l_classifier.cpu().numpy())
    elif args.dataset_name in ["coco", "oid", "lvis"] and args.aggregator != 'plain':

        print('Loading CLIP')
        global_encoder, global_preprocess = clip.load(args.clip_model, device=device)


        gpt_results = load_json(os.path.join(args.gpt_results_root,
                                             f"cleaned_{args.dataset_name}_gpt_detail_hrchy_l1.json"))

        theme_maker = Themer(method=args.aggregator, thresh=args.peigen_thresh, alpha=args.alpha)

        theme_tree_features = defaultdict(dict)

        theme_feat_dict = defaultdict()
        for cat_id, entry in sorted(gpt_results.items(), key=lambda item: int(item[0])):
            if args.dataset_name in ["coco", "lvis"]:
                novel_list = coco_novel if args.dataset_name == "coco" else lvis_novel
                node_sentences = entry["candidate_sentences"] if entry['node_name'] in novel_list else [f"a {entry['node_name']}"]

                candidate_sentences = entry["candidate_sentences"] if entry['node_name'] in novel_list else [f"a {entry['node_name']}"]
                detail_sentences = entry["detail_sentences"] if entry['node_name'] in novel_list else [f"a {entry['node_name']}"]
                sentences_to_use = select_sentences_by_level(candidate_sentences, detail_sentences, 'l1',
                                                             sentence_type='combined')
                truncated_sentences = [sentence[:77] for sentence in sentences_to_use]

            else:
                node_sentences = entry["candidate_sentences"]

            print(f"{entry['node_name']}: {len(truncated_sentences)}")

            node_tokens = clip.tokenize(truncated_sentences).to(device)
            with torch.no_grad():
                node_features = global_encoder.encode_text(node_tokens)
            node_features = F.normalize(node_features)
            # if node_features.size(0) == 1:
            #     print(f"level={level_name} cat={cat_id}")
            #     print(node_sentences)
            #     sys.exit()
            node_theme = theme_maker.get_theme(node_features)
            theme_feat_dict[cat_id] = node_theme

        total_num = len(list(theme_feat_dict.values()))
        print(f"Total feats = {total_num}")

        # Prepare and Save Features
        sorted_theme_dict = OrderedDict(sorted(theme_feat_dict.items(), key=lambda x: int(x[0])))

        l_feats = list(sorted_theme_dict.values())
        l_classifier = torch.stack(l_feats)
        print(f"---> {args.dataset_name}'s classifier has a shape of {l_classifier.shape}")

        # Save the embeddings
        path_save = os.path.join(args.out_path, f"{args.dataset_name}_clip_hrchy_l1.npy")

        print(f'Saving to {path_save}')
        np.save(open(path_save, 'wb'), l_classifier.cpu().numpy())
    elif args.dataset_name in ["coco", "oid", "lvis"] and args.aggregator == 'plain':

        print('Loading CLIP')
        global_encoder, global_preprocess = clip.load(args.clip_model, device=device)

        gpt_results = load_json(os.path.join(args.gpt_results_root,
                                             f"cleaned_{args.dataset_name}_gpt_hrchy_l1.json"))

        class_names = [entry["node_name"] for _, entry in sorted(gpt_results.items(), key=lambda item: int(item[0]))]
        acname_prompts = [f'a {cname}' for cname in class_names]


        classifier_tokens = clip.tokenize(acname_prompts).to(device)

        with torch.no_grad():
            cls_features = global_encoder.encode_text(classifier_tokens)

        l_classifier = F.normalize(cls_features)

        print(f"---> {args.dataset_name}'s classifier has a shape of {l_classifier.shape}")

        # Save the embeddings 生成用來分類的embeddings
        path_save = os.path.join(args.out_path, f"{args.dataset_name}_clip_hrchy_l1.npy")

        print(f'Saving to {path_save}')
        np.save(open(path_save, 'wb'), l_classifier.cpu().numpy())





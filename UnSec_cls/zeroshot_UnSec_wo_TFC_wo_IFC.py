import argparse
import torch
import clip
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import torchmetrics
from data_utils.load_data import load_imagenet_val
from data_utils.cnames_imagenet import IMAGENET_CLASSES
from utils.themer import Themer
from utils.ca import MeanFeatureCalculator2
import json
import sys
import time
from utils.fileios import *
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
DEBUG_MODE = False
COUNT_RELATIVES = False

save_dir="../UnSec-master/MeanFeatureCalculator"
meanFeatureCalculator = MeanFeatureCalculator2(save_dir=save_dir)

inat_layer_policy = {
    'l1': 'no_gm',
    'l2': 'no_gm',
    'l3': 'no_gm',
    'l4': 'no_gm',
    'l5': 'no_gm',
    'l6': 'gm'
    }

def classification_acc(device, top_k: int = 1):
  acc = torchmetrics.Accuracy(task="multiclass",
                              num_classes=1000,
                              top_k=top_k).to(device)
  return acc


def build_clip(model_size: str, device: str, jit: bool):
  # load model
  encoder, preprocesser = clip.load(model_size, device=device, jit=jit)
  encoder.eval()
  encoder.requires_grad_(False)
  return encoder, preprocesser


def do_clip_zeroshot(args, model, dloader, class_names, templates=['a {}']):
  print("=== CLIP Zero-shot ===")

  print("---> Generating classifier")
  zeroshot_weights = []
  for classname in tqdm(class_names):
    texts = [template.format(classname) for template in templates]
    tokens = clip.tokenize(texts).to(args.device)
    txt_embeddings = model.encode_text(tokens)
    txt_embeddings = torch.mean(txt_embeddings, dim=0)
    zeroshot_weights.append(txt_embeddings)

  txt_classifier = torch.stack(zeroshot_weights)
  txt_classifier = F.normalize(txt_classifier)

  print(f"\tclassifier_dim = {txt_classifier.shape}")


  print("---> Evaluating")
  acc_top1 = classification_acc(args.device, top_k=1)
  acc_top5 = classification_acc(args.device, top_k=5)

  total_time = 0.0
  num_images = 1000 if DEBUG_MODE else len(dloader)

  for batch_idx, (images, labels) in enumerate(tqdm(dloader)):
    start_time = time.time()

    images = images.to(args.device)
    labels = labels.to(args.device)

    img_embeddings = model.encode_image(images)
    img_embeddings = F.normalize(img_embeddings)

    scores_clip = img_embeddings @ txt_classifier.T
    preds_clip = scores_clip.argmax(dim=1)
    end_time = time.time()

    acc_clip_ = acc_top1(scores_clip, labels)
    acc_top5_clip_ = acc_top5(scores_clip, labels)

    total_time += (end_time - start_time)
    if DEBUG_MODE and batch_idx >= num_images-1:
      break

  sec_per_item = total_time / num_images
  fps = 1.0 / sec_per_item

  return acc_top1.compute().item(), acc_top5.compute().item(), sec_per_item, fps


# def compute_mean_features(dloader, model, device, model_size, out_path, max_samples=None):
#     """
#     计算数据集的均值特征，并保存到指定路径。
    
#     Args:
#         dloader (DataLoader): 数据加载器，用于遍历数据集
#         model (nn.Module): CLIP 模型
#         device (str): 设备（'cuda' 或 'cpu'）
#         model_size (str): 模型大小（'ViT-B/32', 'ViT-B/16', 'ViT-L/14'）
#         out_path (str): 输出路径，用于保存均值特征
#         max_samples (int, optional): 最大样本数，若达到则提前停止
#     """
    
#     # 解析 model_size 名称
#     model_name = model_size.replace('/', '_')
#     mean_feature_path = os.path.join(out_path, f"{model_name}_mean_features.npy")

#     # 如果均值特征文件已经存在，直接加载并返回
#     if os.path.exists(mean_feature_path):
#         print(f"均值特征文件已存在，直接加载 {mean_feature_path}")
#         return np.load(mean_feature_path)

#     # 存储所有图像的特征
#     all_features = []

#     # 遍历数据加载器中的每个批次
#     for idx, (images, labels) in enumerate(tqdm(dloader, desc="Encoding images")):
#         # 将图像移动到指定设备
#         images = images.to(device)

#         # 提取图像特征
#         with torch.no_grad():
#             img_features = model.encode_image(images)
#             img_features = F.normalize(img_features, p=2, dim=1)  # 归一化特征
#             all_features.append(img_features.cpu().numpy())  # 转为 numpy 并保存

#         # 打印进度信息
#         if idx % 10 == 0:
#             print(f"已编码 {idx} 张图像")

#         # 如果达到最大样本数，则提前退出
#         if max_samples is not None and idx + 1 >= max_samples:
#             print(f"已达到最大样本数 {max_samples}，提前退出")
#             break

#     # 堆叠所有特征并计算均值特征
#     print("开始计算均值特征")
#     all_features = np.vstack(all_features)  # 堆叠为 (N, C)
#     mean_features = np.mean(all_features, axis=0)  # 计算均值特征

#     # 确保 mean_features 的形状与 img_features 一致，调整为 (1, 512)
#     mean_features = mean_features.reshape(1, -1)

#     # 检查 mean_features 和 img_features 的维度是否一致
#     if mean_features.shape != img_features.shape:
#         raise ValueError(f"Error: mean_features shape {mean_features.shape} does not match img_features shape {img_features.shape}")

#     # 保存均值特征文件
#     np.save(mean_feature_path, mean_features)
#     print(f"均值特征已保存至 {mean_feature_path}")

#     return mean_features

import os
import numpy as np
import torch

def compute_mean_features(class_tree, theme_maker, save_dir, dataset_name, model, device="cuda", template='a {}'):
    """
    计算每个类别的均值特征，并将其保存到文件。

    Args:
        class_tree (dict): 类别特征字典，键为类别ID，值为类别特征。
        save_dir (str): 保存均值特征的目录。
        dataset_name (str): 数据集名称，用于区分保存路径。
        model (CLIP model): CLIP模型。
        device (str): 设备类型，默认 "cuda"。  

    Returns:
        mean_features (dict): 每个类别的均值特征。
    """
    dataset_save_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_save_dir, exist_ok=True)

    mean_features = {}

    # 遍历每个类别
    for cat_id, entry in class_tree.items():
        mean_feature_path = os.path.join(dataset_save_dir, f"{cat_id}_mean_features.npy")
        
        # 如果均值特征已经存在，直接加载
        if os.path.exists(mean_feature_path):
            mean_features[cat_id] = torch.tensor(np.load(mean_feature_path))
            print(f"已加载类别 {cat_id} 的均值特征")
            continue  # 跳过当前类别，直接进入下一个类别
        
        # 如果均值特征不存在，计算并保存
        all_features = []
        texts = entry["candidate_sentences"]
        texts = [template.format(t) for t in texts]  # 使用模板处理文本
        tokens = clip.tokenize(texts).to(device)  # 将文本编码为 token 并移动到设备

        # 使用 CLIP 模型进行文本编码
        txt_embeddings = model.encode_text(tokens)
        # 去除维度为1的部分
        all_features.append(txt_embeddings)  # 将生成的文本特征加入 all_features
        
        # 聚合当前类别的所有特征（在这里只有一个特征，txt_embeddings）
        stacked_features = torch.stack(all_features)  # 堆叠特征
        stacked_features = stacked_features.squeeze(0)
        
        # 获取均值特征
        mean_feature = theme_maker.get_theme(stacked_features)

        mean_features[cat_id] = mean_feature

        # 保存均值特征到文件
        np.save(mean_feature_path, mean_feature.cpu().numpy())
        print(f"保存类别 {cat_id} 的均值特征到文件{dataset_save_dir}")

    return mean_features


def compute_global_mean(class_tree, model, device="cuda"):
    """
    计算所有类别特征的全局均值。

    Args:
        class_tree (dict): 类别特征字典，键为类别ID，值为类别特征。
        model (CLIP model): CLIP模型。
        device (str): 设备类型，默认 "cuda"。  

    Returns:
        global_mean (Tensor): 所有类别特征的全局均值。
    """
    all_features = []

    # 遍历所有类别，生成特征
    for cat_id, entry in class_tree.items():
        texts = entry["candidate_sentences"]
        tokens = clip.tokenize(texts).to(device)  # 将文本编码为 token 并移动到设备

        # 使用 CLIP 模型生成文本特征
        txt_embeddings = model.encode_text(tokens)

        # 对特征进行均值处理，使得每个类别有一个固定大小的特征向量
        mean_feature = txt_embeddings.mean(dim=0)  # 对每个类别的特征进行均值计算，得到一个 [512] 的特征向量
        all_features.append(mean_feature)  # 将均值特征加入 all_features

    # 堆叠所有类别的均值特征，并计算全局均值
    stacked_features = torch.stack(all_features)  # 堆叠所有类别的均值特征
    global_mean = stacked_features.mean(dim=0)  # 计算全局均值

    return global_mean


# def correct_domain_bias_iNat(class_tree, mean_features, global_mean=None):
#     """
#     校正类别特征的领域偏置。

#     Args:
#         class_tree (dict): 类别特征字典，键为类别ID，值为类别特征。
#         mean_features (dict): 每个类别的均值特征。
#         global_mean (torch.Tensor, optional): 所有类别的全局均值。

#     Returns:
#         corrected_class_features (dict): 校正后的类别特征。
#     """
#     corrected_class_features = {}

#     for cat_id, entry in class_tree.items():
#         text_features = mean_features.get(cat_id)
#         if text_features is None:
#             print(f"Warning: 类别 {cat_id} 没有均值特征，跳过")
#             continue

#         # 校正领域偏置
#         t_hat = mean_features[cat_id].clone()

#         # 确保 t_hat 和 global_mean 在同一设备上
#         if global_mean is not None:
#             if t_hat.device != global_mean.device:
#                 t_hat = t_hat.to(global_mean.device)  # 将 t_hat 移动到 global_mean 所在设备
        
#             t_hat -= global_mean  # 去除全局偏置

#         # 计算去偏置后的特征
#         text_features = text_features.to(t_hat.device)  
#         centered_text = text_features - t_hat

#         # 确保 centered_text 维度正确，并且是二维的
#         if centered_text.dim() != 2:
#             print(f"Warning: centered_text 的维度不对，实际维度: {centered_text.shape}")
#             # 你可以在此处理维度问题，例如reshape或选择一个合适的维度

#         norm = torch.norm(centered_text, p=2, dim=1, keepdim=True)  # 计算L2范数
#         corrected_text_features = centered_text / (norm + 1e-6)  # 归一化

#         corrected_class_features[cat_id] = corrected_text_features

#     return corrected_class_features

def correct_domain_bias_iNat(zeroshot_weights, mean_features, global_mean=None):
    """
    校正类别特征的领域偏置，包括全局偏置和类别偏置。

    Args:
        zeroshot_weights (list of torch.Tensor): 每个类别的特征列表。
        mean_features (dict): 每个类别的均值特征，用于去除类别偏置。
        global_mean (torch.Tensor, optional): 所有类别的全局均值，用于去除全局偏置。

    Returns:
        corrected_zeroshot_weights (list of torch.Tensor): 校正后的类别特征列表。
    """
    corrected_zeroshot_weights = []

    for cat_id, t_hat in enumerate(zeroshot_weights):
        # 打印维度来调试
        print(f"t_hat.shape for category {cat_id}: {t_hat.shape}")
        
        # 如果 global_mean 存在，先去除全局偏置
        if global_mean is not None:
            # 确保 t_hat 和 global_mean 在同一设备上
            if t_hat.device != global_mean.device:
                t_hat = t_hat.to(global_mean.device)  # 将 t_hat 移动到 global_mean 所在设备
            
            # 去除全局偏置
            t_hat -= global_mean

        # 如果 mean_features 存在，去除类别偏置
        if cat_id in mean_features:
            cat_mean = mean_features[cat_id]
            # 确保 cat_mean 和 t_hat 在同一设备上
            if cat_mean.device != t_hat.device:
                cat_mean = cat_mean.to(t_hat.device)
            
            t_hat -= cat_mean  # 去除类别偏置

        # 如果 t_hat 是一维向量，直接进行归一化
        if len(t_hat.shape) == 1:
            # 计算L2范数，并进行归一化
            norm = torch.norm(t_hat, p=2)  # 计算L2范数
            t_hat = t_hat / (norm + 1e-6)  # 归一化

        corrected_zeroshot_weights.append(t_hat)  # 直接添加校正后的特征

    return corrected_zeroshot_weights


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

def get_combined_theme(global_encoder, theme_maker, candidate_sentences, detail_sentences, device, alpha=0.5):
    """
    Encode candidate and detail sentences separately, get their themes, and combine them.

    Args:
        global_encoder (CLIP model): The CLIP model.
        theme_maker (Themer instance): The Themer instance.
        candidate_sentences (list of str): List of candidate sentences.
        detail_sentences (list of str): List of detail sentences.
        device (str): 'cuda' or 'cpu'.
        alpha (float): Weight for detail_sentences. Default is 0.5.

    Returns:
        Tensor: Combined node theme vector.
    """
    # Encode candidate_sentences
    if candidate_sentences:
        candidate_tokens = clip.tokenize(candidate_sentences).to(device)
        with torch.no_grad():
            candidate_features = global_encoder.encode_text(candidate_tokens)  # [N, C]
        candidate_mean = candidate_features.mean(dim=0, keepdim=True)  # [1, C]
        node_theme_candidate = theme_maker.get_theme(candidate_mean)  # [1, C]
    else:
        # 如果没有 candidate_sentences，使用零向量
        candidate_mean = torch.zeros(1, global_encoder.encode_text(clip.tokenize(['']).to(device)).shape[-1], device=device)
        node_theme_candidate = theme_maker.get_theme(candidate_mean)

    # Encode detail_sentences
    if detail_sentences:
        detail_tokens = clip.tokenize(detail_sentences).to(device)
        with torch.no_grad():
            detail_features = global_encoder.encode_text(detail_tokens)  # [M, C]
        detail_mean = detail_features.mean(dim=0, keepdim=True)  # [1, C]
        node_theme_detail = theme_maker.get_theme(detail_mean)  # [1, C]
    else:
        # 如果没有 detail_sentences，使用零向量
        detail_mean = torch.zeros_like(candidate_mean)
        node_theme_detail = theme_maker.get_theme(detail_mean)

    # Combine themes with weighting
    combined_node_theme = alpha * node_theme_detail + (1 - alpha) * node_theme_candidate  # [1, C]
    return combined_node_theme.squeeze(0)  # [C]


def do_UnSec(args, aggr_method, model, dloader, class_tree, template='a {}'):
    print(f"=== UnSec {aggr_method} ===")
    theme_maker = Themer(method=aggr_method, thresh=1, alpha=0.5)

    print(f"\tGenerating classifier")
    zeroshot_weights = []
    for cat_id, entry in tqdm(sorted(class_tree.items(), key=lambda item: int(item[0]))):
        # texts = entry["candidate_sentences"]+entry["detail_sentences"] # combine
        # texts = entry["detail_sentences"]  # combine
        node_name = entry.get("node_name", None) 
        texts = entry["candidate_sentences"]
        texts = [template.format(t) for t in texts]

        tokens = clip.tokenize(texts).to(args.device)
        # 在这里去除文本的领域偏置
        txt_embeddings = model.encode_text(tokens)
        txt_embeddings = theme_maker.get_theme(txt_embeddings)

        ############################################################################################################
        texts = entry.get("candidate_sentences", [])
        candidate_sentences = [template.format(t) for t in texts]
        detail_sentences = entry.get("detail_sentences", [])

        ####################去除文本领域偏置############################################################

        # current_feature = generate_features(model, texts, args.device, aggregation='mean').clone().detach().to(torch.float32)
        # node_theme = theme_maker.get_theme(txt_embeddings).clone().detach().to(torch.float32)
        node_theme = get_combined_theme(model, theme_maker, texts, detail_sentences, args.device).clone().detach().to(torch.float32)
        node_name_feature = get_node_name_feature(node_name, model, args.device,).clone().detach().to(torch.float32)  # 已经是 float32
        
        ####################SR稀疏重构优化#############################################################
        level_name = 'l1'
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
        else:
            optimized_feature = node_theme
        zeroshot_weights.append(optimized_feature)
    # zeroshot_weights.append(txt_embeddings)
    # mean_features = compute_mean_features(class_tree, theme_maker,save_dir=args.out_path,model=model,dataset_name="imagenet2012", device=args.device)

    # # dloader, model, device, model_size, out_path, max_samples=None
    # global_mean_features = None
    # ########################文本处理#####################################
    # # Step 2: Compute global mean (μ_avg)
    # global_mean = compute_global_mean(class_tree,model,args.device)

    # # corrected_tree_features = correct_domain_bias_iNat0111(theme_tree_features, mean_features, global_mean, inat_layer_policy)

    # correct_zeroshot_weights = correct_domain_bias_iNat(zeroshot_weights, mean_features, global_mean)

    txt_classifier = torch.stack(zeroshot_weights)
    txt_classifier = F.normalize(txt_classifier)

    print(f"\tclassifier_dim = {txt_classifier.shape}")

    print("---> Evaluating")
    acc_top1 = classification_acc(args.device, top_k=1)
    acc_top5 = classification_acc(args.device, top_k=5)
    total_time = 0.0
    num_images = 1000 if DEBUG_MODE else len(dloader)

    for batch_idx, (images, labels) in enumerate(tqdm(dloader)):
        start_time = time.time()

        images = images.to(args.device)
        labels = labels.to(args.device)

        img_embeddings = model.encode_image(images)
        ####################计算均值特征######################################
        # mean_features = meanFeatureCalculator.compute_mean_features_imagenet("imagenet2012",dloader,model)
        # mean_features = compute_mean_image_features(dloader, model, args.device, args.model_size,args.out_path, max_samples=None)

        ####################图像去除领域偏置######################################
        # corrected_backbone_features = meanFeatureCalculator._correct_domain_bias_imagenet2012(img_embeddings,mean_features)

        img_embeddings = F.normalize(img_embeddings)#img_embeddings
        img_embeddings = img_embeddings.to(torch.float32)
        scores_clip = img_embeddings @ txt_classifier.T
        preds_clip = scores_clip.argmax(dim=1)
        end_time = time.time()

        acc_clip_ = acc_top1(scores_clip, labels)# scores_clip 和真实标签算准确率
        acc_top5_clip_ = acc_top5(scores_clip, labels)

        total_time += (end_time - start_time)
        if DEBUG_MODE and batch_idx >= num_images-1:
            break

        sec_per_item = total_time / num_images
        fps = 1.0 / sec_per_item

    return acc_top1.compute().item(), acc_top5.compute().item(), sec_per_item, fps


def parse_k(value):
    """自定义的类型转换函数，将 'all' 转换为 None，否则转换为整数"""
    if value == 'all':
        return 'all'  # 返回 'all' 表示选择所有特征
    try:
        return int(value)  # 如果是数字，转换为整数
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid value for --k, must be 'all' or an integer")
    

def print_results(method, results, hierarchy_source):
  method_name = method.upper()
  print("\n")
  print("=" * 25 + f" {method_name}-based Final Results " + "=" * 25)
  print("\n")
  print(f"[Classification]")
  print(f"Top-1 Acc   : {100 * results[0]}")
  print(f"Top-5 Acc   : {100 * results[1]}")
  print(f"[Speed]")
  print(f"Sec per Item: {results[2]} secs")
  print(f"FPS         : {results[3]} fps")
  print("=" * 25 + "          END          " + "=" * 25)
  output_results = f"top1\ttop5\tSPI\tFPS\n{100 * results[0]},\t{100 * results[1]},\t{results[2]},\t{results[3]}"
  # output_path = method.replace(" ", "_").replace(".", "_").replace("/", "-")
  # output_path = output_path + "_" + hierarchy_source.replace("hierarchy/", "").replace(".json", "")
  # dump_txt(f"output_speed/{output_path}", output_results)
  # print(f"Succ. dumped experiment results to: "+f"output_speed/{output_path}")


import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

def compute_mean_image_features(dloader, model, device, model_size, out_path, max_samples=None):
    """
    计算数据集的均值特征，并保存到指定路径。
    
    Args:
        dloader (DataLoader): 数据加载器，用于遍历数据集
        model (nn.Module): CLIP 模型
        device (str): 设备（'cuda' 或 'cpu'）
        model_size (str): 模型大小（'ViT-B/32', 'ViT-B/16', 'ViT-L/14'）
        out_path (str): 输出路径，用于保存均值特征
        max_samples (int, optional): 最大样本数，若达到则提前停止
    """
    
    # 解析 model_size 名称
    model_name = model_size.replace('/', '_')
    mean_feature_path = os.path.join(out_path, f"{model_name}_mean_features.npy")

    # 如果均值特征文件已经存在，直接加载并返回
    if os.path.exists(mean_feature_path):
        print(f"均值特征文件已存在，直接加载 {mean_feature_path}")
        return np.load(mean_feature_path)

    # 存储所有图像的特征
    all_features = []

    # 遍历数据加载器中的每个批次
    for idx, (images, labels) in enumerate(tqdm(dloader, desc="Encoding images")):
        # 将图像移动到指定设备
        images = images.to(device)

        # 提取图像特征
        with torch.no_grad():
            img_features = model.encode_image(images)
            img_features = F.normalize(img_features, p=2, dim=1)  # 归一化特征
            all_features.append(img_features.cpu().numpy())  # 转为 numpy 并保存

        # 打印进度信息
        if idx % 10 == 0:
            print(f"已编码 {idx} 张图像")

        # 如果达到最大样本数，则提前退出
        if max_samples is not None and idx + 1 >= max_samples:
            print(f"已达到最大样本数 {max_samples}，提前退出")
            break

    # 堆叠所有特征并计算均值特征
    print("开始计算均值特征")
    all_features = np.vstack(all_features)  # 堆叠为 (N, C)
    mean_features = np.mean(all_features, axis=0)  # 计算均值特征

    # 检查 mean_features 和 img_features 的维度是否一致
    # 将 mean_features 的维度从 (512,) 转换为 (1, 512)
    mean_features = mean_features.reshape(1, -1)

    # 检查维度一致性
    if mean_features.shape != img_features.shape:
        raise ValueError(f"Error: mean_features shape {mean_features.shape} does not match img_features shape {img_features.shape}")

    # 保存均值特征文件
    np.save(mean_feature_path, mean_features)
    print(f"均值特征已保存至 {mean_feature_path}")

    return mean_features



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='UnSec_classification',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--dataset_root',
                      type=str,
                      default="./datasets/imagenet2012/",
                      )
  parser.add_argument('--model_size',
                      type=str,
                      default='ViT-B/32',
                      choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
                      )
  parser.add_argument('--method',
                      type=str,
                      default="UnSec",
                      choices=['zeroshot', 'UnSec'],
                      )
  parser.add_argument('--hierarchy_tree_path',
                      type=str,
                      default="./UnSec_cls/hrchy_imagenet1k/imagenet1k_detail_llm_composed.json",
                      choices=[
                          "hrchy_imagenet1k/imagenet1k_hrchy_wordnet.json",         # WordNet hierarchy
                          "hrchy_imagenet1k/imagenet1k_hrchy_llm_composed.json",    # LLM-generated hierarchy
                          "./UnSec_cls/hrchy_imagenet1k/imagenet1k_detail_llm_composed.json",  # 修改了这里
                      ],
  )
  parser.add_argument('--batch_size',
                      type=int,
                      default=1,
                      )
  parser.add_argument('--num_runs',
                      type=int,
                      default=1,
                      )
  parser.add_argument('--enable_global_mean', action='store_true', default=True, help='是否开启多领域总体均值校正') # 使用原本的修正方式前5层不需要开启gm第6层需要,使用最新的一直需要开启
  parser.add_argument('--enable_cross_level_mean', action='store_true', default=True, help='是否开启跨粒度层级的均值校正')
  parser.add_argument('--num_epochs', type=int, default=2, help="Number of epochs for optimization")
  parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for optimization")
  parser.add_argument('--k', type=parse_k, default='all', help="Number of most relevant features to select, or 'all' to select all features")
  parser.add_argument('--optimizer', type=str, default='orin', choices=['orin', 'adm'], help="Select optimizer: 'adam' for traditional gradient descent, 'adm' for ADM")
  parser.add_argument('--out_path', default='./nexus/imagenet2012/vitB32/UnSec_llm_by_level_TFC_0113_new_layer_policy_5n1g_new_epoch2_dgm') # UnSec_llm_by_level_TFC_0109_new_layer_policy_5n1g,UnSec_llm_by_level_TFC_0109_new_layer_policy_6g，UnSec_llm_by_level_TFC_0111_new_layer_policy_5n1g_w_SR_epoch100
    
  args = parser.parse_args()
  args.device = 'cuda'

  encoder, preprocesser = build_clip(args.model_size, args.device, jit=False)
  dset, dloader = load_imagenet_val(dataset_root=args.dataset_root, batch_size=args.batch_size, num_workers=8,
                                    shuffle=False if args.num_runs > 1 else True    # do not shuffle for testing FPS
                                    )

  class_tree = load_json(args.hierarchy_tree_path)
  print(f"Loaded hierarchy tree from: {args.hierarchy_tree_path}")

  # Baseline
  if args.method == "zeroshot":
    num_runs = args.num_runs
    total_fps = 0

    for i in range(num_runs):
      zeroshot_results = do_clip_zeroshot(args, encoder, dloader, IMAGENET_CLASSES, templates=['a {}'])
      total_fps += zeroshot_results[-1]

    zeroshot_results = list(zeroshot_results)
    zeroshot_results[-1] = total_fps / num_runs
    print_results(method=f"CLIP-Zeroshot_w_{args.model_size}", results=zeroshot_results,
                  hierarchy_source=args.hierarchy_tree_path)
  elif args.method == "UnSec":
    num_runs = args.num_runs
    total_fps = 0

    for i in range(num_runs):
      UnSec_mean_results = do_UnSec(args, "mean", encoder, dloader, class_tree, template='a {}')
      total_fps += UnSec_mean_results[-1]

    UnSec_mean_results = list(UnSec_mean_results)
    UnSec_mean_results[-1] = total_fps / num_runs
    print_results(method=f"UnSec-Mean_w_{args.model_size}", results=UnSec_mean_results,
                  hierarchy_source=args.hierarchy_tree_path)
  else:
    raise NotImplementedError(f"Method - {args.method} - is not supported!")



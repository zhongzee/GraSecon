import argparse
import torch
import clip
from torch.nn import functional as F
from tqdm import tqdm
import torchmetrics
from data_utils.load_data import load_imagenet_val
from data_utils.cnames_imagenet import IMAGENET_CLASSES
from utils.themer import Themer
import time
from utils.fileios import *
from utils.ca import MeanFeatureCalculator2
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

save_dir="../GraSecon-master/MeanFeatureCalculator"
meanFeatureCalculator = MeanFeatureCalculator2(save_dir=save_dir)

inat_layer_policy = {
    'l1': 'no_gm',
    'l2': 'no_gm',
    'l3': 'no_gm',
    'l4': 'no_gm',
    'l5': 'no_gm',
    'l6': 'gm'
    }

def classification_acc(device, num_classes, top_k: int = 1):
  acc = torchmetrics.Accuracy(task="multiclass",
                              num_classes=num_classes,
                              top_k=top_k).to(device)
  return acc


def build_clip(model_size: str, device: str, jit: bool):
  # load model
  encoder, preprocesser = clip.load(model_size, device=device, jit=jit)
  encoder.eval()
  encoder.requires_grad_(False)
  return encoder, preprocesser

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

def do_clip_zeroshot(args, model, dloader, class_tree, label_mapper, templates=['a {}']):
  print("=== CLIP Zero-shot ===")
  class_tree.update(sorted(class_tree.items(), key=lambda item: int(item[0])))

  print("---> Generating classifier")
  class_names = [v["node_name"] for _, v in class_tree.items()]

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
  if txt_classifier.size(0) < 5:
    acc_top1 = classification_acc(args.device, num_classes=int(txt_classifier.size(0)), top_k=1)
    acc_top5 = classification_acc(args.device, num_classes=int(txt_classifier.size(0)), top_k=1)
  else:
    acc_top1 = classification_acc(args.device, num_classes=int(txt_classifier.size(0)), top_k=1)
    acc_top5 = classification_acc(args.device, num_classes=int(txt_classifier.size(0)), top_k=5)

  total_time = 0.0
  num_images = 0

  for batch_idx, (og_images, og_labels) in enumerate(tqdm(dloader)):
    og_images = og_images.to(args.device)
    og_labels = og_labels.to(args.device)

    # ADAPTION TO BREEDS HIERARCHY
    # Create a mask for labels that exist in the label_mapper
    mask = torch.tensor([str(label.item()) in label_mapper for label in og_labels])
    # If none of the labels are in the mapper, skip this iteration
    if not mask.any():
      continue
    # Use the mask to get the filtered original images
    images = og_images[mask]
    # Use the mask to get the filtered original labels
    filtered_og_labels = og_labels[mask]
    # Map the original labels to the new ones for the filtered tensor
    labels = torch.tensor([label_mapper[str(label.item())] for label in filtered_og_labels])

    start_time = time.time()

    images = images.to(args.device)
    labels = labels.to(args.device)

    img_embeddings = model.encode_image(images)
    img_embeddings = F.normalize(img_embeddings)

    scores_clip = img_embeddings @ txt_classifier.T
    preds_clip = scores_clip.argmax(dim=1)

    acc_clip_ = acc_top1(scores_clip, labels)
    acc_top5_clip_ = acc_top5(scores_clip, labels)

    end_time = time.time()
    total_time += (end_time - start_time)
    num_images += labels.size(0)

  sec_per_item = total_time / num_images
  fps = 1.0 / sec_per_item

  return acc_top1.compute().item(), acc_top5.compute().item(), sec_per_item, fps

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

import torch
import torch.nn.functional as F

def transform_correct_zeroshot_weights(correct_theme_tree_features, level):
    """
    将校正后的特征转换为与原始zeroshot_weights相同的结构。
    
    Args:
        correct_theme_tree_features (dict): 校正后的层级特征字典，形状为 {level_name: {id: Tensor}}。
        level (str): 当前层级的名称，例如 'l1', 'l2' 等。
    
    Returns:
        torch.Tensor: 校正后的零样本分类权重，形状为 [num_categories, feature_dim]。
    """
    correct_zeroshot_weights_list = []

    # 按照类别ID的顺序遍历校正后的特征
    for cat_id in sorted(correct_theme_tree_features[level].keys()):
        corrected_embedding = correct_theme_tree_features[level][cat_id]
        
        # 如果校正特征是列表形式，合并成一个张量
        if isinstance(corrected_embedding, list):
            corrected_embedding = torch.stack(corrected_embedding)

        # 添加到结果列表
        correct_zeroshot_weights_list.append(corrected_embedding)

    # 将所有校正后的特征堆叠成一个张量
    # correct_zeroshot_weights = torch.stack(correct_zeroshot_weights_list)  # [num_categories, feature_dim]
    
    return correct_zeroshot_weights_list

def get_combined_theme(global_encoder, theme_maker, candidate_sentences, detail_sentences, device, level=None, alpha=0.5):
    """
    Encode candidate and detail sentences separately, get their themes, and combine them.
    
    Args:
        global_encoder (CLIP model): The CLIP model.
        theme_maker (Themer instance): The Themer instance.
        candidate_sentences (list of str): List of candidate sentences.
        detail_sentences (list of str): List of detail sentences.
        device (str): 'cuda' or 'cpu'.
        level (str): The current level (e.g., 'L1', 'L2', etc.).
        alpha (float): Weight for detail_sentences. Default is 0.5.
        
    Returns:
        Tensor: Combined node theme vector.
    """

    # print("当前层级是",level)
    if level in ['l4', 'l5', 'l6']:
        alpha = 0.18  #
    
    # Encode candidate_sentences
    if candidate_sentences:
        candidate_tokens = clip.tokenize(candidate_sentences).to(device)
        with torch.no_grad():
            candidate_features = global_encoder.encode_text(candidate_tokens)  # [N, C]
        candidate_mean = candidate_features.mean(dim=0, keepdim=True)  # [1, C]
        node_theme_candidate = theme_maker.get_theme(candidate_mean)  # [1, C]
    else:
        # If there are no candidate_sentences, use a zero vector
        candidate_mean = torch.zeros(1, global_encoder.encode_text(clip.tokenize(['']).to(device)).shape[-1], device=device)
        node_theme_candidate = theme_maker.get_theme(candidate_mean)

    # Encode detail_sentences
    if detail_sentences:
        # Truncate sentences if they are longer than 77 tokens
        truncated_sentences = [sentence[:77] if len(sentence) > 77 else sentence for sentence in detail_sentences]
        
        # Tokenize the sentences, using the truncated sentences if necessary
        detail_tokens = clip.tokenize(truncated_sentences).to(device)
        with torch.no_grad():
            detail_features = global_encoder.encode_text(detail_tokens)  # [M, C]
        detail_mean = detail_features.mean(dim=0, keepdim=True)  # [1, C]
        node_theme_detail = theme_maker.get_theme(detail_mean)  # [1, C]
    else:
        # If there are no detail_sentences, use a zero vector
        detail_mean = torch.zeros_like(candidate_mean)
        node_theme_detail = theme_maker.get_theme(detail_mean)

    # Combine themes with weighting (alpha determines the weight of detail_sentences)
    combined_node_theme = alpha * node_theme_detail + (1 - alpha) * node_theme_candidate  # [1, C]
    return combined_node_theme.squeeze(0)  # [C]



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
    # 检查特征维度是否一致
    if mean_features.shape[-1] != img_features.shape[-1]:
        raise ValueError(f"Error: mean_features feature dimension {mean_features.shape[-1]} does not match img_features feature dimension {img_features.shape[-1]}")

    # 保存均值特征文件
    np.save(mean_feature_path, mean_features)
    print(f"均值特征已保存至 {mean_feature_path}")

    return mean_features

def do_GraSecon(args, aggr_method, model, dloader, class_tree, label_mapper):
    print(f"=== GraSecon {aggr_method} ===")
    theme_maker = Themer(method=aggr_method, thresh=1, alpha=0.5)

    # 如果是多层级，逐个层级处理
    if args.breed_level == 'all':
        for level, tree in class_tree.items():
            print(f"\nProcessing level {level}")
            level_results = process_level(args, model, dloader, tree, label_mapper, theme_maker,level)
            print_results(f"GraSecon_Mean_{args.model_size}_BREEDS_{level}", level_results)
    else:
        print(f"\tGenerating classifier")
        # 只有单一层级时处理
        level_results = process_level(args, model, dloader, class_tree, label_mapper, theme_maker, args.breed_level)
        print_results(f"GraSecon_Mean_{args.model_size}_BREEDS_{args.breed_level}", level_results)


from collections import defaultdict

def process_level(args, model, dloader, class_tree, label_mapper, theme_maker,level):
    zeroshot_weights = []
    theme_tree_features = defaultdict(dict)
    for cat_id, entry in class_tree.items():
        node_name = entry['node_name']
        texts = entry["candidate_sentences"]
        detail_sentences = entry["detail_sentences"]
        tokens = clip.tokenize(texts).to(args.device)
        txt_embeddings = model.encode_text(tokens)
        txt_embeddings = theme_maker.get_theme(txt_embeddings)
        
        node_theme = get_combined_theme(model, theme_maker, texts, detail_sentences, args.device, level).clone().detach().to(torch.float32)

        # zeroshot_weights.append(node_theme) # 
        zeroshot_weights.append(node_theme) # 
        theme_tree_features[level][cat_id] = node_theme

    # mean_features = text_calculator.compute_mean_features_2(args.dataset_name, all_text_features, sentence_type=args.sentence_type)
    mean_features = compute_mean_features(theme_tree_features, save_dir=args.out_path, dataset_name="breeds", device=args.device)
    global_mean_features = None

    # Step 2: Compute global mean (μ_avg)
    global_mean = compute_global_mean(theme_tree_features)
    
    # corrected_tree_features = correct_domain_bias_iNat0111(theme_tree_features, mean_features, global_mean, inat_layer_policy)
    
    correct_theme_tree_features = correct_domain_bias_iNat(theme_tree_features, mean_features, global_mean, inat_layer_policy)

    correct_zeroshot_weights = transform_correct_zeroshot_weights(correct_theme_tree_features, level)
     # 输出：[num_categories, feature_dim]，例如 [10, 512]

    txt_classifier = torch.stack(zeroshot_weights)#zeroshot_weights
    txt_classifier = F.normalize(txt_classifier)

    print(f"\tclassifier_dim = {txt_classifier.shape}")

    acc_top1 = classification_acc(args.device, num_classes=int(txt_classifier.size(0)), top_k=1)
    acc_top5 = classification_acc(args.device, num_classes=int(txt_classifier.size(0)), top_k=5)

    total_time = 0.0
    num_images = 0

    for batch_idx, (og_images, og_labels) in enumerate(tqdm(dloader)):
        og_images = og_images.to(args.device)
        og_labels = og_labels.to(args.device)

        # ADAPTION TO BREEDS HIERARCHY
        mask = torch.tensor([str(label.item()) in label_mapper for label in og_labels])
        if not mask.any():
            continue

        images = og_images[mask]
        filtered_og_labels = og_labels[mask]
        labels = torch.tensor([label_mapper[str(label.item())] for label in filtered_og_labels])

        ####################计算均值特征######################################
        # mean_features = compute_mean_image_features(dloader, model, args.device, args.model_size, args.out_path, max_samples=None)

        start_time = time.time()

        images = images.to(args.device)
        labels = labels.to(args.device)

        img_embeddings = model.encode_image(images)

        ####################图像去除领域偏置######################################
        # corrected_backbone_features = meanFeatureCalculator._correct_domain_bias_imagenet2012(img_embeddings, mean_features)

        img_embeddings = F.normalize(img_embeddings)
        img_embeddings = img_embeddings.to(torch.float32)
        scores_clip = img_embeddings @ txt_classifier.T
        preds_clip = scores_clip.argmax(dim=1)

        acc_clip_ = acc_top1(scores_clip, labels)
        acc_top5_clip_ = acc_top5(scores_clip, labels)

        end_time = time.time()
        total_time += (end_time - start_time)
        num_images += labels.size(0)

    sec_per_item = total_time / num_images
    fps = 1.0 / sec_per_item

    return acc_top1.compute().item(), acc_top5.compute().item(), sec_per_item, fps


# def do_GraSecon(args, aggr_method, model, dloader, class_tree, label_mapper):
#   print(f"=== GraSecon {aggr_method} ===")
#   theme_maker = Themer(method=aggr_method, thresh=1, alpha=0.5)
#   class_tree.update(sorted(class_tree.items(), key=lambda item: int(item[0])))

#   print(f"\tGenerating classifier")
#   zeroshot_weights = []
#   for cat_id, entry in class_tree.items():
#     texts = entry["candidate_sentences"]
#     tokens = clip.tokenize(texts).to(args.device)
#     txt_embeddings = model.encode_text(tokens)
#     txt_embeddings = theme_maker.get_theme(txt_embeddings)
#     zeroshot_weights.append(txt_embeddings)

#   mean_features = compute_mean_features(class_tree, theme_maker,save_dir=args.out_path,model=model,dataset_name="imagenet2012", device=args.device)

#   mean_features = None
#   # dloader, model, device, model_size, out_path, max_samples=None
#   global_mean_features = None
#   ########################文本处理#####################################
#   # Step 2: Compute global mean (μ_avg)
#   global_mean = compute_global_mean(class_tree,model,args.device)

#   # corrected_tree_features = correct_domain_bias_iNat0111(theme_tree_features, mean_features, global_mean, inat_layer_policy)

#   correct_zeroshot_weights = correct_domain_bias_iNat(zeroshot_weights, mean_features, global_mean)

#   txt_classifier = torch.stack(correct_zeroshot_weights)
#   txt_classifier = F.normalize(txt_classifier)

#   print(f"\tclassifier_dim = {txt_classifier.shape}")


#   print("---> Evaluating")
#   if txt_classifier.size(0) < 5:
#     acc_top1 = classification_acc(args.device, num_classes=int(txt_classifier.size(0)), top_k=1)
#     acc_top5 = classification_acc(args.device, num_classes=int(txt_classifier.size(0)), top_k=1)
#   else:
#     acc_top1 = classification_acc(args.device, num_classes=int(txt_classifier.size(0)), top_k=1)
#     acc_top5 = classification_acc(args.device, num_classes=int(txt_classifier.size(0)), top_k=5)

#   total_time = 0.0
#   num_images = 0

#   for batch_idx, (og_images, og_labels) in enumerate(tqdm(dloader)):
#     og_images = og_images.to(args.device)
#     og_labels = og_labels.to(args.device)

#     # ADAPTION TO BREEDS HIERARCHY
#     # Create a mask for labels that exist in the label_mapper
#     mask = torch.tensor([str(label.item()) in label_mapper for label in og_labels])
#     # If none of the labels are in the mapper, skip this iteration
#     if not mask.any():
#       continue
#     # Use the mask to get the filtered original images
#     images = og_images[mask]
#     # Use the mask to get the filtered original labels
#     filtered_og_labels = og_labels[mask]
#     # Map the original labels to the new ones for the filtered tensor
#     labels = torch.tensor([label_mapper[str(label.item())] for label in filtered_og_labels])

#     start_time = time.time()

#     images = images.to(args.device)
#     labels = labels.to(args.device)

#     img_embeddings = model.encode_image(images)
#     ####################计算均值特征######################################
#     # mean_features = meanFeatureCalculator.compute_mean_features_imagenet("imagenet2012",dloader,model)
#     mean_features = compute_mean_image_features(dloader, model, args.device, args.model_size,args.out_path, max_samples=None)

#     ####################图像去除领域偏置######################################
#     corrected_backbone_features = meanFeatureCalculator._correct_domain_bias_imagenet2012(img_embeddings,mean_features)

#     img_embeddings = F.normalize(corrected_backbone_features)#img_embeddings
    
#     # img_embeddings = F.normalize(img_embeddings)

#     scores_clip = img_embeddings @ txt_classifier.T
#     preds_clip = scores_clip.argmax(dim=1)

#     acc_clip_ = acc_top1(scores_clip, labels)
#     acc_top5_clip_ = acc_top5(scores_clip, labels)

#     end_time = time.time()
#     total_time += (end_time - start_time)
#     num_images += labels.size(0)

#   sec_per_item = total_time / num_images
#   fps = 1.0 / sec_per_item

#   return acc_top1.compute().item(), acc_top5.compute().item(), sec_per_item, fps


def print_results(method, results):
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
  # output_results = f"top1\ttop5\tSPI\tFPS\n{100 * results[0]},\t{100 * results[1]},\t{results[2]},\t{results[3]}"
  # output_path = method.replace(" ", "_").replace(".", "_").replace("/", "-")
  # dump_txt(f"output_breeds/{output_path}", output_results)
  # print(f"Succ. dumped experiment results to: "+f"output_speed/{output_path}")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='GraSecon_classification',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--dataset_root',
                      type=str,
                      default="./datasets/imagenet2012/",
                      )
  parser.add_argument('--model_size',
                      type=str,
                      default='ViT-B/16',
                      choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
                      )
  parser.add_argument('--method',
                      type=str,
                      default="GraSecon",
                      choices=['zeroshot', 'GraSecon']
                      )
  parser.add_argument('--breed_level',
                      type=str,
                      default='l6',
                      choices=['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'all'],
                      help="选择一个层级或'all'来加载所有层级")

  parser.add_argument('--hierarchy_root',
                      type=str,
                      default="./GraSecon_cls/hrchy_breeds",
                      )
  parser.add_argument('--batch_size',
                      type=int,
                      default=64,
                      )
  
  parser.add_argument('--enable_global_mean', action='store_true', default=True, help='是否开启多领域总体均值校正') # 使用原本的修正方式前5层不需要开启gm第6层需要,使用最新的一直需要开启
  parser.add_argument('--enable_cross_level_mean', action='store_true', default=True, help='是否开启跨粒度层级的均值校正')
  parser.add_argument('--num_epochs', type=int, default=2, help="Number of epochs for optimization")
  parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for optimization")
  # parser.add_argument('--k', type=parse_k, default='all', help="Number of most relevant features to select, or 'all' to select all features")
  parser.add_argument('--optimizer', type=str, default='orin', choices=['orin', 'adm'], help="Select optimizer: 'adam' for traditional gradient descent, 'adm' for ADM")
  parser.add_argument('--out_path', default='./nexus/imagenet2012/vitB32/hrchy_breeds') # GraSecon_llm_by_level_TFC_0109_new_layer_policy_5n1g,GraSecon_llm_by_level_TFC_0109_new_layer_policy_6g，GraSecon_llm_by_level_TFC_0111_new_layer_policy_5n1g_w_SR_epoch100
    


  args = parser.parse_args()
  args.device = 'cuda'

  encoder, preprocesser = build_clip(args.model_size, args.device, jit=False)
  dset, dloader = load_imagenet_val(dataset_root=args.dataset_root, batch_size=args.batch_size, num_workers=8,
                                    shuffle=True)

  hier_paths = {
    'l1': f"{args.hierarchy_root}/composed_details_breed_l2_num_class=10.json",
    'l2': f"{args.hierarchy_root}/composed_details_breed_l3_num_class=29.json",
    'l3': f"{args.hierarchy_root}/composed_details_breed_l4_num_class=128.json",
    'l4': f"{args.hierarchy_root}/composed_details_breed_l5_num_class=466.json",
    'l5': f"{args.hierarchy_root}/composed_details_breed_l6_num_class=591.json",
    'l6': f"{args.hierarchy_root}/composed_details_breed_l7_num_class=98.json",
  }

  hier_mapper_paths = {
    'l1': f"{args.hierarchy_root}/mapper_l2_leaf2current.json",
    'l2': f"{args.hierarchy_root}/mapper_l3_leaf2current.json",
    'l3': f"{args.hierarchy_root}/mapper_l4_leaf2current.json",
    'l4': f"{args.hierarchy_root}/mapper_l5_leaf2current.json",
    'l5': f"{args.hierarchy_root}/mapper_l6_leaf2current.json",
    'l6': f"{args.hierarchy_root}/mapper_l7_leaf2current.json",
  }

  # class_tree = load_json(hier_paths[args.breed_level])
  if args.breed_level == 'all':
    class_tree = {}
    for level in ['l1', 'l2', 'l3', 'l4', 'l5', 'l6']:
        class_tree[level] = load_json(hier_paths[level])
        label_mapper = load_json(hier_mapper_paths[level])
  else:
    class_tree = load_json(hier_paths[args.breed_level])
    label_mapper = load_json(hier_mapper_paths[args.breed_level])
 

  # Baseline
  if args.method == "zeroshot":
    zeroshot_results = do_clip_zeroshot(args, encoder, dloader, class_tree, label_mapper, templates=['a {}'])
    print_results(
      method=f"CLIP-Zeroshot_baseline_{args.model_size}_BREEDS_{args.breed_level}", results=zeroshot_results,
    )
  # GraSecon
  elif args.method == "GraSecon":
    GraSecon_mean_results = do_GraSecon(args, "mean", encoder, dloader, class_tree, label_mapper)

    # print_results(
    #   method=f"GraSecon_Mean_{args.model_size}_BREEDS_{args.breed_level}", results=GraSecon_mean_results
    # )
  else:
    raise NotImplementedError(f"Method - {args.method} - is not supported!")



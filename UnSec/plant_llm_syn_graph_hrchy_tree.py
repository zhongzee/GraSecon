import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import os
import clip
from collections import defaultdict, OrderedDict
from copy import deepcopy
from UnSec.tools.llm_controllers import LLMBot
from UnSec.tools.fileios import *
from UnSec.tools.themer import Themer
import networkx as nx
from generate_graph_candidate_sentences import generate_graph_candidate_sentences

# Function to build graph and update gpt_results with similarity data
def build_graph_from_results(results_dict, embeddings, child_embeddings, top_k_similar=5, top_k_dissimilar=5):
    updated_results_dict = {}

    for node_id, node_info in results_dict.items():
        node_embedding = embeddings[node_id]  # torch.Size([512])
        similarities = []
        child_similarities = []

        # Compare this node embedding to all others
        for other_node_id, other_node_info in results_dict.items():
            if node_id != other_node_id:
                other_embedding = embeddings[other_node_id]
                similarity = cosine_similarity(node_embedding, other_embedding) # 余弦相似度
                similarities.append((other_node_id, similarity))

        # Sort similarities
        similarities.sort(key=lambda x: x[1], reverse=True)
        # 取最相似的前 top_k_similar 个节点作为 hard negative nodes
        top_similar = [(results_dict[x[0]]['node_name'], float(x[1])) for x in similarities[:top_k_similar]]

        # 取最不相似的前 top_k_dissimilar 个节点作为 easy negative nodes
        top_dissimilar = [(results_dict[x[0]]['node_name'], float(x[1])) for x in similarities[-top_k_dissimilar:]]

        # 计算当前节点与其所有子节点的相似度
        for child_name in node_info.get('child_names', []):
            child_id = f"{node_id}_child_{child_name}"
            if child_id in child_embeddings:
                child_similarity = cosine_similarity(node_embedding, child_embeddings[child_id])
                child_similarities.append((child_name, child_similarity))

        # Update node_info with similar and dissimilar nodes and their similarity values
        node_info['hard_negative_nodes_with_scores'] = [(name, float(score)) for name, score in top_similar]
        node_info['easy_negative_nodes_with_scores'] = [(name, float(score)) for name, score in top_dissimilar]
        node_info['child_similarities_with_scores'] = [(name, float(score)) for name, score in child_similarities]

        # Save the updated node_info to a new dictionary
        updated_results_dict[node_id] = node_info

    return updated_results_dict

def cosine_similarity(embedding1, embedding2):
    embedding1 = embedding1.cpu().numpy()  # 将 Tensor 移动到 CPU 并转换为 NumPy 数组
    embedding2 = embedding2.cpu().numpy()  # 将 Tensor 移动到 CPU 并转换为 NumPy 数组
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def build_graph_with_embeddings(args, level_names, device, global_encoder, theme_maker):
    theme_tree_features = defaultdict(dict)
    child_embeddings = {}  # 用于存储所有子节点的嵌入

    for level_name in level_names:
        theme_tree_features_file = os.path.join(args.out_path, f"{args.dataset_name}_theme_tree_features_{level_name}.npy")
        child_embeddings_file = os.path.join(args.out_path, f"{args.dataset_name}_child_embeddings_{level_name}.npy")

        # 如果已经存在嵌入文件，加载它们
        if os.path.exists(theme_tree_features_file) and os.path.exists(child_embeddings_file):
            print(f"Loading saved embeddings for level {level_name}...")
            theme_tree_features[level_name] = np.load(theme_tree_features_file, allow_pickle=True).item()
            child_embeddings = np.load(child_embeddings_file, allow_pickle=True).item()
            # Load hierarchy results
            gpt_results = load_json(
                os.path.join(args.gpt_results_root, f"cleaned_{args.dataset_name}_gpt_hrchy_{level_name}.json")
            )

        else:
            # Load hierarchy results
            gpt_results = load_json(
                os.path.join(args.gpt_results_root, f"cleaned_{args.dataset_name}_gpt_hrchy_{level_name}.json")
            )

            # Process embeddings for each category using CLIP
            for cat_id, entry in gpt_results.items():
                # 获取当前节点的嵌入
                node_sentences = entry["candidate_sentences"]
                node_tokens = clip.tokenize(node_sentences).to(device)

                with torch.no_grad():
                    node_features = global_encoder.encode_text(node_tokens)

                node_features = F.normalize(node_features)
                node_theme = theme_maker.get_theme(node_features)
                theme_tree_features[level_name][cat_id] = node_theme

                # 计算子节点的嵌入
                for i, child_name in enumerate(entry.get('child_names', [])):
                    if i < len(entry['candidate_sentences']):
                        child_sentence = entry['candidate_sentences'][i]
                        child_tokens = clip.tokenize([child_sentence]).to(device)

                        with torch.no_grad():
                            child_features = global_encoder.encode_text(child_tokens)

                        child_features = F.normalize(child_features)
                        child_id = f"{cat_id}_child_{child_name}"
                        child_embeddings[child_id] = child_features.squeeze(dim=0)

            # 保存嵌入以供将来使用
            np.save(theme_tree_features_file, theme_tree_features[level_name])
            np.save(child_embeddings_file, child_embeddings)

        num_classes = len(gpt_results)
        if level_name == 'l1' and num_classes <= args.top_k_similar:
            top_k_similar = num_classes // 2  # 将top_k_similar设置为类目数量的一半
            top_k_dissimilar = num_classes - top_k_similar -1
        else:
            top_k_similar = args.top_k_similar
            top_k_dissimilar = args.top_k_dissimilar

        # 构建图结构并加入相似/不相似节点的连接
        gpt_results = build_graph_from_results(gpt_results, theme_tree_features[level_name], child_embeddings, top_k_similar, top_k_dissimilar)

        args.output_path = os.path.join(args.out_path, f"cleaned_{args.dataset_name}_gpt_graph_hrchy_{level_name}")

        graph_results = generate_graph_candidate_sentences(gpt_results)

        dump_json(args.output_path, graph_results)

    return gpt_results  # 返回包含相似和不相似节点的图结构


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='fsod',
                        choices=['inat', 'fsod', 'coco', 'lvis', 'oid'])
    parser.add_argument('--gpt_results_root', default='fsod_llm_answers')
    parser.add_argument('--prompter', default='isa', choices=['a', 'avg', 'concat', 'isa'])
    parser.add_argument('--aggregator', default='mean', choices=['peigen', 'mean', 'mixed', 'all_eigens', 'plain'])
    parser.add_argument('--peigen_thresh', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--clip_model', default="ViT-B/32", choices=['ViT-B/32', 'RN50'])
    parser.add_argument('--out_path', default='fsod_llm_graph_answers')
    parser.add_argument('--top_k_similar', type=int, default=5)
    parser.add_argument('--top_k_dissimilar', type=int, default=5)
    parser.add_argument('--generate_graph_results', action='store_true', help='generate_graph_results')

    args = parser.parse_args()

    # # Ensure output folder exists
    # if not is_valid_folder(args.out_path):
    #     raise FileExistsError

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP model
    print('Loading CLIP')
    global_encoder, global_preprocess = clip.load(args.clip_model, device=device)

    # Setup Themer
    theme_maker = Themer(method=args.aggregator if args.aggregator != "plain" else "mean", thresh=args.peigen_thresh,
                         alpha=args.alpha)

    # Define dataset levels
    if args.dataset_name == 'inat':
        level_names = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']
        # level_names = ['l1']
    else:
        level_names = ['l3', 'l2', 'l1']

    # Build graph with embeddings and update similarity/dissimilarity nodes
    graph_results = build_graph_with_embeddings(args, level_names, device, global_encoder, theme_maker)
    # print("graph_results=",graph_results)
    # print("out_path=",args.out_path)
    # args.output_path = os.path.join(args.output_path, f"raw_{args.dataset_name}_gpt_graph_hrchy_{args.level_names}")
    # # Save the updated results with similar and dissimilar nodes in JSON format
    # dump_json(args.output_path, graph_results)

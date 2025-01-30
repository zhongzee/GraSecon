import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import os
import clip
from collections import defaultdict, OrderedDict
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from UnSec.tools.fileios import load_json, dump_json
import networkx as nx

# 修改后的graph构建部分，支持父子、兄弟、跨层级关系
def build_graph_with_gnn(results_dict, embeddings, child_embeddings, top_k_similar=5, top_k_dissimilar=5):
    graph = nx.Graph()
    node_features = {}

    # Step 1: 添加节点及其特征
    for node_id, node_info in results_dict.items():
        node_embedding = embeddings[node_id]
        graph.add_node(node_id)
        node_features[node_id] = node_embedding

        # 添加子节点，并添加父子关系边
        for child_name in node_info.get('child_names', []):
            child_id = f"{node_id}_child_{child_name}"
            if child_id in child_embeddings:
                child_embedding = child_embeddings[child_id]
                graph.add_node(child_id)
                graph.add_edge(node_id, child_id)  # 添加父子节点间的边
                node_features[child_id] = child_embedding

        # Step 2: 添加兄弟节点关系（同一层级的节点）
        for sibling_id in results_dict.keys():
            if sibling_id != node_id:
                graph.add_edge(node_id, sibling_id)  # 添加兄弟节点间的边

    # Step 3: 添加跨层级关系（如相似节点）
    updated_results_dict = {}
    for node_id, node_info in results_dict.items():
        node_embedding = embeddings[node_id]
        similarities = []
        for other_node_id, other_node_info in results_dict.items():
            if node_id != other_node_id:
                other_embedding = embeddings[other_node_id]
                similarity = cosine_similarity(node_embedding, other_embedding)
                similarities.append((other_node_id, similarity))

        # 按相似度排序并添加相似/不相似的节点
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar = similarities[:top_k_similar]
        top_dissimilar = similarities[-top_k_dissimilar:]

        # Step 4: 更新节点信息，加入相似和不相似节点
        node_info['similar_nodes'] = [x[0] for x in top_similar]
        node_info['dissimilar_nodes'] = [x[0] for x in top_dissimilar]
        updated_results_dict[node_id] = node_info

        # 为最相似的节点添加边
        for sim_id, _ in top_similar:
            graph.add_edge(node_id, sim_id)

    return graph, node_features, updated_results_dict


# 计算余弦相似度
def cosine_similarity(embedding1, embedding2):
    embedding1 = embedding1.cpu().numpy()
    embedding2 = embedding2.cpu().numpy()
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


# GNN模型定义
class GCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 128)
        self.conv2 = GCNConv(128, 64)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# 主函数：构建全局图并进行特征聚合
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载 CLIP 模型
    print('Loading CLIP')
    global_encoder, global_preprocess = clip.load(args.clip_model, device=device)

    # 设置数据集层级与路径
    level_names = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']
    embeddings, child_embeddings = {}, {}

    # 跨层级构建图结构
    for level_name in level_names:
        gpt_results_path = os.path.join(args.gpt_results_root, f"cleaned_{args.dataset_name}_gpt_hrchy_{level_name}.json")
        gpt_results = load_json(gpt_results_path)

        # Step 1: 编码 candidate sentences
        for node_id, entry in gpt_results.items():
            node_sentences = entry["candidate_sentences"]
            node_tokens = clip.tokenize(node_sentences).to(device)
            with torch.no_grad():
                node_features = global_encoder.encode_text(node_tokens)
            node_features = F.normalize(node_features)
            embeddings[node_id] = node_features.mean(dim=0)  # 对候选句子特征取平均

            # 编码子节点
            for child_name in entry.get('child_names', []):
                child_sentence = f"{child_name}, which is a {entry['node_name']}"
                child_tokens = clip.tokenize([child_sentence]).to(device)
                with torch.no_grad():
                    child_features = global_encoder.encode_text(child_tokens)
                child_features = F.normalize(child_features)
                child_id = f"{node_id}_child_{child_name}"
                child_embeddings[child_id] = child_features.squeeze(dim=0)

        # Step 2: 构建图结构并计算相似性
        graph, node_features, gpt_results = build_graph_with_gnn(gpt_results, embeddings, child_embeddings)

        # Step 3: 准备 PyTorch Geometric 数据对象用于 GNN
        edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
        x = torch.stack(list(node_features.values()))
        data = Data(x=x, edge_index=edge_index)

        # Step 4: 训练 GCN 进行特征聚合
        model = GCN(num_node_features=x.size(1)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        data = data.to(device)
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out, data.x)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}: Loss {loss.item()}")

        # Step 5: 保存 GNN 增强的节点特征为npy文件，每个层级存一个大的npy文件
        enhanced_node_features = out.cpu().detach().numpy()
        np.save(os.path.join(args.out_path, f"{args.dataset_name}_gnn_hrchy_{level_name}.npy"), enhanced_node_features)

        # 保存图关系和结果
        dump_json(os.path.join(args.out_path, f"cleaned_{args.dataset_name}_graph_hrchy_{level_name}.json"), gpt_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='inat', choices=['inat', 'fsod'])
    parser.add_argument('--gpt_results_root', default='inat_llm_answers')
    parser.add_argument('--clip_model', default="ViT-B/32", choices=['ViT-B/32', 'RN50'])
    parser.add_argument('--out_path', default='./nexus/inat/vitB32/UnSec_llm/')
    args = parser.parse_args()

    main(args)

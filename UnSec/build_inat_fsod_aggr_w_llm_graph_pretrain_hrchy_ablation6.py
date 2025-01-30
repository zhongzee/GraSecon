import argparse
import os
import json
import torch
import numpy as np
from collections import defaultdict, OrderedDict
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn.functional as F

# GCN 模型定义
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

# 预训练的GraphSAGE 模型
class PretrainedGraphSAGE(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=128, output_dim=64):
        super(PretrainedGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 加载预训练的GNN模型
def load_pretrained_gnn(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 读取npy文件
def load_npy_files(npy_root, levels):
    npy_data = {}
    for level in levels:
        npy_path = os.path.join(npy_root, f'inat_clip_hrchy_{level}.npy')
        npy_data[level] = np.load(npy_path)
        print(f"Loaded {npy_path} with shape {npy_data[level].shape}")
    return npy_data

# 读取json文件并提取层次关系
def load_json_hierarchy(json_root, levels):
    json_data = {}
    for level in levels:
        json_path = os.path.join(json_root, f'cleaned_inat_gpt_hrchy_{level}.json')
        with open(json_path, 'r') as f:
            json_data[level] = json.load(f)
    return json_data

# 构建层次图结构
def build_graph_and_features(json_data, npy_data, levels):
    graph = defaultdict(list)
    features = {}

    for level in levels:
        level_data = json_data[level]
        for node_id, node_info in level_data.items():
            features[node_id] = npy_data[level][int(node_id) - 1]  # 假设json文件的id与npy文件行号对应

            # 建立父子节点的连接
            for child_name in node_info.get('child_names', []):
                child_node_id = get_node_id_by_name(child_name, json_data[level])  # 查找子节点id
                if child_node_id:
                    graph[node_id].append(child_node_id)
    return graph, features

# 根据名称查找节点ID
def get_node_id_by_name(name, level_data):
    for node_id, node_info in level_data.items():
        if node_info['node_name'] == name:
            return node_id
    return None

# 使用GNN进行特征汇聚
# 使用 GNN 进行特征汇聚（含预训练模型的逻辑）
def aggregate_features_with_gnn(graph, features, device, args):
    nodes = list(features.keys())
    node_indices = {node: i for i, node in enumerate(nodes)}

    # 构建边列表
    edge_list = []
    for node, children in graph.items():
        for child in children:
            edge_list.append([node_indices[node], node_indices[child]])

    # 转换为tensor格式
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)
    x = torch.stack([torch.tensor(features[node]) for node in nodes]).to(device)

    # 构建数据对象
    data = Data(x=x, edge_index=edge_index)

    # 判断是否使用预训练模型
    if args.use_pretrained_gnn:
        if args.pretrained_model_path is None:
            raise ValueError("Must provide --pretrained_model_path when using --use_pretrained_gnn")

        print(f"Loading pretrained GNN from {args.pretrained_model_path}")
        model = PretrainedGraphSAGE(num_node_features=x.size(1)).to(device)
        model.load_state_dict(torch.load(args.pretrained_model_path))
    else:
        print("Training a new GNN model.")
        model = PretrainedGraphSAGE(num_node_features=x.size(1)).to(device)

        # 初始化优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()

        for epoch in range(50):
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out, data.x)  # 监督训练，目标为保留原特征
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}: Loss {loss.item()}")

    # 返回汇聚后的特征
    model.eval()
    with torch.no_grad():
        aggregated_features = model(data).cpu().detach().numpy()

    return aggregated_features


# 主函数
# 主函数
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    levels = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1'] if args.dataset_name == 'inat' else ['l3', 'l2', 'l1']

    # 1. 加载npy特征
    npy_data = load_npy_files(args.npy_root, levels)

    # 2. 加载json文件提取层次关系
    json_data = load_json_hierarchy(args.gpt_results_root, levels)

    # 3. 构建层次图结构和节点特征
    graph, features = build_graph_and_features(json_data, npy_data, levels)

    # 4. 使用GNN进行特征汇聚（包含预训练模型逻辑）
    aggregated_features = aggregate_features_with_gnn(graph, features, device, args)

    # 5. 保存聚合后的特征
    for level in levels:
        output_path = os.path.join(args.out_path, f'{args.dataset_name}_gnn_aggregated_{level}.npy')
        np.save(output_path, aggregated_features)
        print(f"Saved aggregated features for {level} to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='inat', choices=['inat', 'fsod'])
    parser.add_argument('--gpt_results_root', default='inat_llm_answers')
    parser.add_argument('--npy_root', default='./nexus/inat/vitB32/UnSec_llm/')
    parser.add_argument('--out_path', default='./nexus/inat/vitB32/UnSec_gnn_aggregated/')
    parser.add_argument('--use_pretrained_gnn', action='store_true', help='Use pre-trained GNN instead of training one')
    parser.add_argument('--pretrained_model_path', default=None, help='Path to the pre-trained GNN model')
    args = parser.parse_args()

    main(args)

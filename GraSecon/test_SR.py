def generate_features(global_encoder, sentence_list, device, aggregation='mean'):
    """
    使用 VLM 模型生成句子的特征表示，并进行聚合
    """
    tokens = clip.tokenize(sentence_list).to(device)
    with torch.no_grad():
        features = global_encoder.encode_text(tokens).float()  # [num_sentences, feature_dim]
    # 聚合特征
    if aggregation == 'mean':
        aggregated_feature = features.mean(dim=0)  # [feature_dim]
    elif aggregation == 'max':
        aggregated_feature, _ = features.max(dim=0)
    elif aggregation == 'weighted_mean':
        # 示例：根据句子长度或其他标准定义权重
        weights_list = torch.ones(features.shape[0], device=device)  # 需要根据实际情况定义
        aggregated_feature = (features * weights_list.unsqueeze(1)).sum(dim=0) / weights_list.sum()
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation}")
    return aggregated_feature

def get_node_name_feature(node_name, global_encoder, device):
    """
    获取 node_name 的 CLIP 文本特征。
    """
    with torch.no_grad():
        tokens = clip.tokenize([node_name]).to(device)  # 单句
        node_feature = global_encoder.encode_text(tokens).float()  # [1, feature_dim]
    return node_feature.squeeze(0)  # [feature_dim]

def optimize_feature_with_grad_descent(current_feature, node_name_feature, num_epochs=100, lr=0.01):
    """
    使用传统梯度下降法优化 current_feature 使其逼近 node_name_feature。
    """
    current_feature = current_feature.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([current_feature], lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = criterion(current_feature, node_name_feature)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
            try:
                wandb.log({'Loss': loss.item(), 'Epoch': epoch + 1})
            except wandb.Error as e:
                print(f"W&B logging failed: {e}. Skipping log for this step.")

    return current_feature.detach()

def test_single_node():
    # 假设有一个节点的特征
    node_name = "liquid"
    candidate_sentences = [
        "a tea, which is a liquid, which is a matter",
        "a juice, which is a liquid, which is a matter",
        # ... 更多句子 ...
    ]

    # 生成 current_feature
    current_feature = generate_features(global_encoder, candidate_sentences, device, aggregation='mean')
    theme_tree_features[level_name][cat_id] = current_feature.cpu().detach()

    # 获取 node_name_feature
    node_name_feature = get_node_name_feature(node_name, global_encoder, device)

    # 优化
    optimized_feature = optimize_feature_with_adm(
        current_feature=current_feature.to(device),
        node_name_feature=node_name_feature.to(device),
        lambda_val=0.1,
        rho=1.0,
        num_epochs=100,
        lr=0.01,
        early_stop_patience=10
    )

    # 比较优化前后的特征
    similarity_before = F.cosine_similarity(current_feature, node_name_feature, dim=0).item()
    similarity_after = F.cosine_similarity(optimized_feature, node_name_feature, dim=0).item()
    print(f"Cosine Similarity before optimization: {similarity_before:.4f}")
    print(f"Cosine Similarity after optimization: {similarity_after:.4f}")

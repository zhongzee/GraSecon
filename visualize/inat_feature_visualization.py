import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import os
import pandas as pd

# 定义可视化函数
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import os
import seaborn as sns

# 定义可视化函数
def plot_embeddings(embeddings, labels, method="pca", level_name="Level", model_name="Model", out_dir="./output"):
    if method == "pca":
        reducer = PCA(n_components=3)
        title = f'PCA Visualization of {level_name} - {model_name}'
    elif method == "tsne":
        perplexity = min(30, len(embeddings) // 3)  # 动态调整perplexity，确保小于样本数量
        reducer = TSNE(n_components=3, random_state=42, perplexity=perplexity, n_iter=1000)
        title = f't-SNE Visualization of {level_name} - {model_name}'
    else:
        raise ValueError(f"Unknown method: {method}")

    reduced_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    # 使用 Seaborn 调色板，根据标签分配颜色
    unique_labels = np.unique(labels)
    palette = sns.color_palette("hsv", len(unique_labels))
    label_colors = {label: palette[i] for i, label in enumerate(unique_labels)}

    # 根据标签颜色绘制散点图
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=[label_colors[label] for label in labels], alpha=0.7)
    plt.colorbar(scatter, label='Classes', ticks=range(len(unique_labels)))
    plt.title(title)

    # 保存图片
    save_path = os.path.join(out_dir, f"{level_name}_{model_name}_{method}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()




# 定义计算中心点距离的函数并量化
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

def calculate_centroid_distances(embeddings, labels, distance_metric="euclidean", level_name="Level",
                                 model_name="Model", baseline_name="Baseline", out_dir="./output"):
    unique_labels = np.unique(labels)
    centroids = []
    for label in unique_labels:
        class_embeddings = embeddings[labels == label]
        centroid = np.mean(class_embeddings, axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)

    if distance_metric == "euclidean":
        dist_matrix = euclidean_distances(centroids)
    elif distance_metric == "cosine":
        dist_matrix = cosine_distances(centroids)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    # 计算平均距离
    mean_dist = np.mean(dist_matrix[np.triu_indices_from(dist_matrix, k=1)])

    # 打印并保存特征距离和平均距离
    save_path = os.path.join(out_dir, f"{baseline_name}_{level_name}_{model_name}_{distance_metric}_centroid_distances.txt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(f"Centroid Distance Matrix ({distance_metric}) for {level_name} - {model_name} - {baseline_name}:\n")
        f.write(np.array2string(dist_matrix, separator=', '))
        f.write(f"\n\nMean Distance: {mean_dist}\n")

    # 打印日志信息
    log_entry = f"[INFO] {baseline_name} - {level_name} - {model_name} - {distance_metric} Mean Distance: {mean_dist}"
    print(log_entry)

    return mean_dist, log_entry




# 定义保存 Excel 表格的 DataFrame
results_df = pd.DataFrame(columns=["Baseline", "Model", "Level", "Distance Metric", "Mean Distance"])

# 循环处理所有的 baseline, model 和 level
level_names = [f"l{i}" for i in range(1, 7)]  # l1 到 l6
baselines = ["baseline", "UnSec_gt", "UnSec_llm"]
clip_models = ["rn50", "vitB32"]
output_root = "./inat"  # 结果保存路径

for baseline in baselines:
    for model in clip_models:
        for level_name in level_names:
            # 加载 npy 文件
            npy_path = f'.././nexus/inat/{model}/{baseline}/inat_clip_hrchy_{level_name}.npy'
            embeddings = np.load(npy_path)

            # 生成 labels (这里只是生成了简单的连续的数字，可以根据实际数据进行调整)
            num_classes = embeddings.shape[0]
            labels = np.arange(num_classes)

            # 定义保存路径
            output_dir = os.path.join(output_root, model, baseline)

            # 可视化 (PCA, t-SNE)
            plot_embeddings(embeddings, labels, method="pca", level_name=level_name, model_name=model,
                            out_dir=output_dir)
            plot_embeddings(embeddings, labels, method="tsne", level_name=level_name, model_name=model,
                            out_dir=output_dir)

            # 计算并保存特征中心点距离，保存欧式距离和余弦相似度的结果
            for metric in ["euclidean", "cosine"]:
                mean_distance = calculate_centroid_distances(embeddings, labels, distance_metric=metric,
                                                             level_name=level_name, model_name=model,baseline_name=baseline,
                                                             out_dir=output_dir)

                # 初始化空的 DataFrame
                results_df = pd.DataFrame(columns=["Model", "Baseline", "Level", "Distance_Metric", "Mean_Distance"])

                # 将每一行的结果添加到 DataFrame 中，改用 pd.concat
                results_df = pd.concat([results_df, pd.DataFrame({
                    "Model": [model],
                    "Baseline": [baseline],
                    "Level": [level_name],
                    "Distance_Metric": [metric],
                    "Mean_Distance": [mean_distance]
                })], ignore_index=True)

# 保存结果为 Excel 文件
excel_output_path = os.path.join(output_root, "centroid_distances_summary.xlsx")
results_df.to_excel(excel_output_path, index=False)

import json
import os

def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """保存 JSON 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def merge_detail_sentences(base_dir_a, base_dir_b, output_dir, levels):
    # 确保输出路径存在
    os.makedirs(output_dir, exist_ok=True)

    for level in levels:
        # 构造文件路径
        file_a_path = os.path.join(base_dir_a, f'cleaned_inat_gpt_hrchy_{level}.json')
        file_b_path = os.path.join(base_dir_b, f'cleaned_inat_gpt_detail_hrchy_{level}.json')
        output_path = os.path.join(output_dir, f'cleaned_inat_gpt_hrchy_{level}.json')

        # 加载文件 A 和 B
        json_a = load_json(file_a_path)
        json_b = load_json(file_b_path)

        # 遍历文件 B，将 detail_sentences 匹配到文件 A 的相应 node_name 条目
        for node_id, node_data in json_b.items():
            node_name = node_data.get("node_name")
            detail_sentences = node_data.get("detail_sentences", [])
            # candidate_sentences = node_data.get("candidate_sentences", [])

            # 查找文件 A 中具有相同 node_name 的条目并添加 detail_sentences
            for item in json_a.values():
                if item.get("node_name") == node_name:
                    item["detail_sentences"] = detail_sentences
                    # item["candidate_sentences"] = candidate_sentences
                    break

        # 保存合并后的文件
        save_json(json_a, output_path)
        print(f"已处理层级 {level}: 合并结果保存到 {output_path}")

# 参数设置
base_dir_a = 'inat_llm_answers'                        # 文件 A 的基础路径
base_dir_b = 'inat_llm_detail_answers'         # 文件 B 的基础路径
output_dir = 'inat_llm_answers'                 # 输出路径
levels = ['l1', 'l2', 'l3', 'l4', 'l5', 'l6']          # 层级列表
# levels = ['l1', 'l2', 'l3']          # 层级列表

# 调用合并函数
merge_detail_sentences(base_dir_a, base_dir_b, output_dir, levels)

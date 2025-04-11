import json
import argparse
import re

# 设置参数解析器
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='inat', choices=['inat', 'fsod_expanded', 'coco', 'lvis', 'oid'])
parser.add_argument('--description_file', default='raw_inat_gpt_hrchy_l1.json', help='Name of the fine-grained descriptions JSON file')
args = parser.parse_args()

# 设置文件路径
dataset_name = args.dataset_name
description_file_name = args.description_file
description_file_path = f'{dataset_name}_llm_answers_detail_context/{description_file_name}'

hierarchical_file_path = f'{dataset_name}_llm_answers/cleaned_{dataset_name}_gpt_hrchy_l1.json'
output_file_path = f'{dataset_name}_llm_answers/cleaned_{dataset_name}_gpt_detail_context_hrchy_l1.json'

# 读取两个 JSON 文件
with open(description_file_path, 'r') as f1:
    fine_grained_descriptions = json.load(f1)

with open(hierarchical_file_path, 'r') as f2:
    hierarchical_structure = json.load(f2)

# 去除括号内容并忽略大小写的匹配函数
def clean_name(name):
    return re.sub(r'\s*\(.*?\)', '', name).strip().lower()

# 遍历第二个文件，匹配类别并添加细粒度描述
unmatched_categories = []
for key, value in hierarchical_structure.items():
    if isinstance(value, dict):  # 检查 value 是否为字典
        node_name = value.get("node_name")
        if node_name:
            node_name_cleaned = clean_name(node_name)
            matched = False
            for description_key in fine_grained_descriptions:
                description_key_cleaned = clean_name(description_key.split(",")[0])
                if node_name_cleaned == description_key_cleaned:
                    description = fine_grained_descriptions[description_key]
                    if "detail_sentences" not in value:
                        value["detail_sentences"] = []
                    value["detail_sentences"].append(description)
                    matched = True
                    break
            if not matched:
                unmatched_categories.append(node_name)

# 将修改后的内容写回到输出文件中
with open(output_file_path, 'w') as f2_updated:
    json.dump(hierarchical_structure, f2_updated, indent=4)

# 如果有未匹配的类别，抛出错误并打印未匹配的类别
if unmatched_categories:
    raise ValueError(f"未匹配的类别: {unmatched_categories}")

print(f"细粒度描述已成功添加到第二个文件中，并保存为 {output_file_path}。")

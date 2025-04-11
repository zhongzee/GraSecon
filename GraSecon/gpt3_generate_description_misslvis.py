import openai
import time
import json
import argparse
import os

# 设置参数解析器，添加默认值
parser = argparse.ArgumentParser()
parser.add_argument('--api_base', default='xx', help='API base URL for OpenAI.')
parser.add_argument('--api_key', default='xx', help='API key for OpenAI.')
parser.add_argument('--input_file', default='./GraSecon/miss_lvis_oid_llm_answers', help='Input JSON directory containing hierarchical structure.')
parser.add_argument('--output_file', default='./GraSecon/miss_lvis_oid_llm_answers', help='Output JSON directory to store the results.')
parser.add_argument('--level', nargs='+', default=['l3'], choices=['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'all'], help='Specify which level(s) (l1-l6) or all levels to process.')
parser.add_argument('--start_from', default='', help='Specify the class name to start from.')
args = parser.parse_args()

# 设置 API 凭据
openai.api_base = args.api_base
openai.api_key = args.api_key

def text_prompt(class_name, all_classnames):
    other_classnames = [cn for cn in all_classnames if cn != class_name]
    return f"""
    For zero-shot learning and open-world object detection, succinctly describe '{class_name}' focusing on its distinguishing visual features compared to its all similar classes. 
    Include at least 3 distinct attributes in the description. Avoid any irrelevant, meaningless descriptions. Answer as concisely and accurately as possible. No more than 50 words.
    Please describe mainly the external visual features rather than the internal structure.
    There is no need to use examples in the form of 1, 2, 3, just describe them directly. Don't write any nonsense, strictly follow the format below and list the common/special visual features.
    For example: Car is a four-wheeled motor vehicle primarily designed for transportation, distinguishable by its streamlined body shape, presence of doors and windows, and the characteristic presence of headlights and taillights.
    """.strip()

# 从 OpenAI API 生成描述
def generate_description(class_name, all_classnames):
    prompt = text_prompt(class_name, all_classnames)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=70,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        res_content = response['choices'][0]['message']['content']
        return res_content
    except Exception as e:
        print(f"Error generating description for {class_name}: {str(e)}")
        return None

# 保存 JSON 文件的函数
def dump_json(filename: str, in_data):
    if not filename.endswith('.json'):
        filename += '.json'

    with open(filename, 'w') as fbj:
        if isinstance(in_data, dict):
            json.dump(in_data, fbj, indent=4)
        elif isinstance(in_data, list):
            json.dump(in_data, fbj)
        else:
            raise TypeError(f"in_data has wrong data type {type(in_data)}")

# 读取和处理特定层级的 JSON 文件
def process_levels(levels):
    unmatched_categories = []
    log_file_path = 'process_log.json'

    # 如果旧的日志文件存在，读取日志
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as log_file:
            process_log = json.load(log_file)
    else:
        process_log = {}

    for level in levels:
        input_file_path = os.path.join(args.input_file, f'cleaned_oid_lvis_gpt_hrchy_l3.json')
        output_file_path = os.path.join(args.output_file, f'cleaned_oid_lvis_gpt_detail_hrchy_l3.json')

        # 检查输入文件是否存在
        if not os.path.exists(input_file_path):
            print(f"输入文件 {input_file_path} 不存在，跳过...")
            continue

        # 确保输出文件的目录存在
        output_dir = os.path.dirname(output_file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 读取输入 JSON 文件
        with open(input_file_path, 'r') as file:
            hierarchical_structure = json.load(file)

        # 获取所有的节点名称
        all_classnames = [value.get("node_name") for key, value in hierarchical_structure.items() if isinstance(value, dict) and value.get("node_name")]

        # 设置是否开始处理的标志
        start_processing = not args.start_from

        from tqdm import tqdm

        # Initialize the progress bar
        total_nodes = len(hierarchical_structure)
        progress_bar = tqdm(total=total_nodes, desc="Generating descriptions", unit="node")

        # 遍历 JSON 数据，生成描述并添加到 detail_sentences 中
        # Process each node
        for node_id, value in hierarchical_structure.items():
            if isinstance(value, dict):
                node_name = value.get("node_name")

                if node_name:
                    # Check if we should start processing
                    if args.start_from and not start_processing:
                        if node_name == args.start_from:
                            start_processing = True
                        else:
                            continue

                    # Print the node_id along with the node_name
                    print(f"Generating description for Node ID {node_id}: {node_name}...")
                    description = generate_description(node_name, all_classnames)

                    if description:
                        print(f"Description for Node ID {node_id} ({node_name}): {description}")

                        if "detail_sentences" not in value:
                            value["detail_sentences"] = []
                        value["detail_sentences"].append(description)

                        # Save progress in real-time
                        dump_json(output_file_path, hierarchical_structure)
                    else:
                        print(f"Failed to generate description for Node ID {node_id} ({node_name})")
                        unmatched_categories.append(node_name)

                    # Update the progress bar
                    progress_bar.update(1)

        # Close the progress bar after completion
        progress_bar.close()

        # 如果有未匹配的类别，打印它们
        if unmatched_categories:
            print(f"未匹配的类别: {unmatched_categories}")

        print(f"细粒度描述已成功添加到文件中，并保存为 {output_file_path}。")

# 检查哪些节点还没有 detail_sentences
def check_missing_details(levels):
    missing_details = {}

    for level in levels:
        input_file_path = os.path.join(args.output_file, f'./GraSecon/miss_lvis_oid_llm_answers/cleaned_oid_lvis_gpt_detail_hrchy_l3.json')

        # 检查输入文件是否存在
        if not os.path.exists(input_file_path):
            print(f"输入文件 {input_file_path} 不存在，跳过...")
            continue

        # 读取输入 JSON 文件
        with open(input_file_path, 'r') as file:
            hierarchical_structure = json.load(file)

        # 查找没有 detail_sentences 的节点
        for key, value in hierarchical_structure.items():
            if isinstance(value, dict):
                node_name = value.get("node_name")
                if node_name and "detail_sentences" not in value:
                    if level not in missing_details:
                        missing_details[level] = []
                    missing_details[level].append(node_name)

    # 打印没有 detail_sentences 的节点
    for level, nodes in missing_details.items():
        print(f"层级 {level} 中以下节点没有 detail_sentences: {nodes}")
    return missing_details

# 处理并生成缺失描述的节点
# 处理并生成缺失描述的节点，确保不会覆盖已有的 detail_sentences
def process_missing_details(missing_details):
    log_file_path = 'process_log.json'

    # 如果旧的日志文件存在，读取日志
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as log_file:
            process_log = json.load(log_file)
    else:
        process_log = {}

    unmatched_categories = []

    for level, nodes in missing_details.items():
        input_file_path = os.path.join(args.input_file, f'cleaned_oid_lvis_gpt_hrchy_{level}.json')
        output_file_path = os.path.join(args.output_file, f'cleaned_oid_lvis_gpt_detail_hrchy_{level}.json')

        # 检查输入文件是否存在
        if not os.path.exists(input_file_path):
            print(f"输入文件 {input_file_path} 不存在，跳过...")
            continue

        # 读取输入 JSON 文件
        with open(input_file_path, 'r') as file:
            hierarchical_structure = json.load(file)

        # 获取所有的节点名称
        all_classnames = [value.get("node_name") for key, value in hierarchical_structure.items() if isinstance(value, dict) and value.get("node_name")]

        from tqdm import tqdm

        # Initialize the progress bar
        total_nodes = len(nodes)
        progress_bar = tqdm(total=total_nodes, desc="Generating missing descriptions", unit="node")

        # 读取已有的 output 文件，避免覆盖已存在的 detail_sentences
        if os.path.exists(output_file_path):
            with open(output_file_path, 'r') as file:
                hierarchical_structure_output = json.load(file)
        else:
            hierarchical_structure_output = {}

        # 遍历缺失描述的节点，并为每个节点生成描述
        for node_name in nodes:
            # 遍历所有节点，找到对应的节点
            for node_id, value in hierarchical_structure.items():
                if isinstance(value, dict) and value.get("node_name") == node_name:
                    # 如果该节点已经有了 detail_sentences，就跳过
                    if "detail_sentences" in value:
                        print(f"Node {node_name} already has detail_sentences, skipping.")
                        continue
                    
                    # 生成描述
                    print(f"Generating description for missing Node: {node_name}...")
                    description = generate_description(node_name, all_classnames)

                    if description:
                        print(f"Description for Node ({node_name}): {description}")

                        # 确保该节点存在 detail_sentences 并且添加描述
                        if "detail_sentences" not in value:
                            value["detail_sentences"] = []
                        value["detail_sentences"].append(description)

                        # 在原始结构中更新该节点的 detail_sentences
                        hierarchical_structure_output[node_id] = value

                        # 保存进度
                        dump_json(output_file_path, hierarchical_structure_output)
                    else:
                        print(f"Failed to generate description for Node ({node_name})")
                        unmatched_categories.append(node_name)

                    # 更新进度条
                    progress_bar.update(1)

        # 关闭进度条
        progress_bar.close()

        # 如果有未匹配的类别，打印它们
        if unmatched_categories:
            print(f"未匹配的类别: {unmatched_categories}")

        print(f"缺失的细粒度描述已成功添加到文件中，并保存为 {output_file_path}。")

# 处理指定的层级
if 'all' in args.level:
    levels_to_process = ['l1', 'l2', 'l3', 'l4', 'l5', 'l6']
else:
    levels_to_process = args.level

# 调用处理函数
process_levels(levels_to_process)

# 检查哪些节点还没有 detail_sentences
missing_details = check_missing_details(levels_to_process)

# 调用处理缺失描述的函数
process_missing_details(missing_details)


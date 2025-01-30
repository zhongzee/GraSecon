import openai
import time
import json
import argparse
import os
import random  # 用于随机选择

# 设置参数解析器，添加默认值
parser = argparse.ArgumentParser()
parser.add_argument('--api_base', default='https://openkey.cloud/v1', help='API base URL for OpenAI.')
parser.add_argument('--api_key', default='xx',
                    help='API key for OpenAI.')
parser.add_argument('--input_file', default='inat_llm_answers',
                    help='Input JSON directory containing hierarchical structure.')
parser.add_argument('--output_file', default='inat_llm_detail_answers_attributes',
                    help='Output JSON directory to store the results.')
parser.add_argument('--level', nargs='+', default=['l1', 'l2'], choices=['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'all'],
                    help='Specify which level(s) (l1-l6) or all levels to process.')
parser.add_argument('--start_from', default='', help='Specify the class name to start from.')
args = parser.parse_args()

# 设置 API 凭据
openai.api_base = args.api_base
openai.api_key = args.api_key


# 生成候选句子的函数，使用 GPT 动态生成父类和子类的描述

#    Example output should look like:
#         "a {child_name}, which has [specific trait 1], [specific trait 2], and [specific trait 3], which is a {class_name} with [common trait 1], [common trait 2], and [common trait 3]."
#
def text_prompt_candidate_sentences(class_name, parent_name, child_names):
    candidate_sentences = []

    # 如果没有父节点，生成一个更广泛的替代描述
    if not parent_name:
        parent_name = "a broader category"  # 或者可以是 "the kingdom of life" 或 "other broad classifications"

    # 如果没有子节点，生成一个通用的子节点描述
    if not child_names:
        child_names = ["related subclasses"]  # 可以是 "other smaller related groups" 或 "related categories"

    # 遍历所有子类
    for child_name in child_names:
        # 生成提示词
        prompt = f"""
        For zero-shot learning and open-world object detection, succinctly generate detailed descriptions for '{class_name}', focusing on its general visual features, and briefly mention its relationship to '{child_name}'.
        The description should begin with {class_name}, followed by a brief mention of {child_name}, and conclude with its broader classification under {parent_name}.
        Avoid irrelevant, meaningless descriptions. Please describe mainly the external visual features rather than the internal structure. No more than 70 words.
        Please strictly follow this structure:
        Start with the distinguishing features of {class_name}, followed by a mention of {child_name}, and finally indicate that it belongs to {parent_name}.

        Example: "Amphibia, recognized by its moist skin, webbed feet, and dual life cycle, includes species such as Hynobiidae, characterized by its unique adaptations to water and land. Amphibia belongs to Chordata, known for its bilateral symmetry and segmented body."
        """.strip()

        try:
            # 调用 GPT 生成描述
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=80,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            res_content = response['choices'][0]['message']['content']
            candidate_sentences.append(res_content)

        except Exception as e:
            print(f"Error generating description for {child_name}: {str(e)}")

    return candidate_sentences

def text_prompt_chordata_example(class_name, parent_name=None, child_names=None):
    # 定义变量用于存储最终的描述
    subclass_intro = ""

    # 判断子类的情况，最多取3个子类
    if child_names:
        if len(child_names) > 3:
            major_subclasses = ', '.join(random.sample(child_names, 3))
        else:
            major_subclasses = ', '.join(child_names)

    # 根据 parent_name 和 child_names 的存在情况，构造 subclass_intro
    if parent_name and child_names:
        subclass_intro = f"{class_name} includes major subclasses like {major_subclasses} and belongs to {parent_name}. "
    elif parent_name:
        subclass_intro = f"{class_name} belongs to {parent_name}. "
    elif child_names:
        subclass_intro = f"{class_name} includes major subclasses like {major_subclasses}. "

    # 保留原始提示的生成逻辑
    prompt = f"""
    For zero-shot learning and open-world object detection, generate a concise description for '{class_name}' focusing on two aspects: common traits and secondary traits. 

    1. **Common Traits**: Describe the primary traits of '{class_name}' using external visual features where possible. Avoid irrelevant, meaningless descriptions. Keep it concise and focused, no more than 70 words.

    2. **Secondary Traits**: Include secondary traits that are still visible but may vary across the lifecycle or specific subclasses (e.g., presence of a tail beyond the anus, segmented body, or body symmetry).
   
    Avoid irrelevant or complex phrases, and focus on visual features. Please use 'includes' to connect these traits concisely.
    
    Example output: Chordata's common traits includes a notochord, dorsal nerve cord, and pharyngeal slits. Secondary traits includes a post-anal tail and a segmented body structure.
    """.strip()

    return subclass_intro + prompt
# 生成描述
def generate_description(class_name, parent_name, child_names):
    # 调用 text_prompt_candidate_sentences 来生成候选句子
    candidate_sentences = text_prompt_candidate_sentences(class_name, parent_name, child_names)
    return candidate_sentences


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
        input_file_path = os.path.join(args.input_file, f'cleaned_inat_gpt_hrchy_{level}.json')
        output_file_path = os.path.join(args.output_file, f'cleaned_inat_gpt_detail_hrchy_{level}.json')

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
        all_classnames = [value.get("node_name") for key, value in hierarchical_structure.items() if
                          isinstance(value, dict) and value.get("node_name")]

        # 设置是否开始处理的标志
        start_processing = not args.start_from

        # 遍历 JSON 数据，生成描述并添加到 detail_sentences 中
        for key, value in hierarchical_structure.items():
            if isinstance(value, dict):
                node_name = value.get("node_name")
                parent_name = value.get("parent_names", [None])
                child_names = value.get("child_names", [])

                if node_name:
                    # 如果指定了 start_from，找到该节点后再开始处理
                    if args.start_from and not start_processing:
                        if node_name == args.start_from:
                            start_processing = True
                        else:
                            continue

                    # 开始生成描述
                    print(f"Generating description for {node_name}...")
                    description = generate_description(node_name, parent_name, child_names)
                    if description:
                        print(f"Description for {node_name} is {description}")
                        if "detail_sentences" not in value:
                            value["detail_sentences"] = []
                        value["detail_sentences"].extend(description)  # 存储多个生成的句子

                        # 实时保存已处理的节点
                        dump_json(output_file_path, hierarchical_structure)
                    else:
                        print(f"Failed to generate description for {node_name}")
                        unmatched_categories.append(node_name)

        # 如果有未匹配的类别，打印它们
        if unmatched_categories:
            print(f"未匹配的类别: {unmatched_categories}")

        print(f"细粒度描述已成功添加到文件中，并保存为 {output_file_path}。")


# 检查哪些节点还没有 detail_sentences
def check_missing_details(levels):
    missing_details = {}

    for level in levels:
        input_file_path = os.path.join(args.output_file, f'cleaned_inat_gpt_detail_hrchy_{level}.json')

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


# 处理指定的层级
if 'all' in args.level:
    levels_to_process = ['l1', 'l2', 'l3', 'l4', 'l5', 'l6']
else:
    levels_to_process = args.level

# 调用处理函数
process_levels(levels_to_process)

# 检查哪些节点还没有 detail_sentences
check_missing_details(levels_to_process)

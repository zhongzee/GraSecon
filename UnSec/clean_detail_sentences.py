import json
import os


def process_json_files(input_dir, levels):
    # 遍历每个层级的 JSON 文件
    for level in levels:
        input_file_path = os.path.join(input_dir, f'cleaned_inat_gpt_detail_hrchy_{level}.json')

        # 检查文件是否存在
        if not os.path.exists(input_file_path):
            print(f"文件 {input_file_path} 不存在，跳过...")
            continue

        # 读取并处理 JSON 文件
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 遍历 JSON 数据的每个条目
        for key, entry in data.items():
            # 检查是否有 detail_sentences
            if "detail_sentences" in entry:
                # 处理 detail_sentences 列表
                processed_sentences = []
                for sentence in entry["detail_sentences"]:
                    # 查找 "It differs from" 的位置并截取
                    if "They can be differentiated" in sentence:
                        sentence = sentence.split("They can be differentiated")[0].strip()
                    processed_sentences.append(sentence)
                # 更新 detail_sentences
                entry["detail_sentences"] = processed_sentences

        # 保存处理后的 JSON 文件
        with open(input_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"Processed file for level {level}: {input_file_path}")


# 定义输入文件目录和级别列表
input_dir = 'inat_llm_detailed_answers'  # 替换为实际的输入文件夹路径
levels = ['l1', 'l2', 'l3', 'l4', 'l5', 'l6']  # 级别列表

# 调用函数处理所有层级文件
process_json_files(input_dir, levels)

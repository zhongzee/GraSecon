import json
import re
import os

# 读取并处理 JSON 文件的函数
def clean_detail_sentences(input_file, output_file):
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"输入文件 {input_file} 不存在。")
        return

    # 读取输入 JSON 文件
    with open(input_file, 'r') as file:
        data = json.load(file)

    # 正则表达式，用于查找 "It can be visually differentiated from" 及之后的内容
    pattern = re.compile(r"They can be differentiated.*")

    # 遍历每个节点并处理 detail_sentences
    for key, value in data.items():
        if isinstance(value, dict) and "detail_sentences" in value:
            cleaned_sentences = []
            for sentence in value["detail_sentences"]:
                # 使用正则表达式进行替换
                cleaned_sentence = re.sub(pattern, "", sentence).strip()
                cleaned_sentences.append(cleaned_sentence)
            # 更新 detail_sentences
            value["detail_sentences"] = cleaned_sentences

    # 将修改后的内容写回到输出文件中
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"文件已处理并保存为 {output_file}")

# 示例用法
input_file = 'cleaned_inat_gpt_detail_hrchy_l6_2.json'  # 输入文件路径
output_file = 'cleaned_inat_gpt_detail_hrchy_l6_cleaned.json'  # 输出文件路径
clean_detail_sentences(input_file, output_file)

import openai
import time
import json


def generate_descriptions(texts, api_base, api_key):
    openai.api_base = api_base
    openai.api_key = api_key

    def text_prompt(class_name, all_classnames):
        other_classnames = [cn for cn in all_classnames if cn != class_name]
        return f"""
        For zero-shot learning and open-world object detection, succinctly describe '{class_name}' focusing on its distinguishing visual features compared to its all similar classes. 
        Include at least 3 distinct attributes in the description. Avoid any irrelevant, meaningless descriptions. Answer as concisely and accurately as possible. No more than 50 words.
        Please describe mainly the external visual features rather than the internal structure.
        There is no need to use examples in the form of 1, 2, 3, just describe them directly. 
        For example: Car is a four-wheeled motor vehicle primarily designed for transportation, distinguishable by its streamlined body shape, presence of doors and windows, and the characteristic presence of headlights and taillights.
        """.strip()

    def clean_responce(text):
        import re
        pattern = r'\[.*?\]'
        match = re.search(pattern, text)
        if match:
            inner_text = match.group()
            inner_text = inner_text[1:-1]  # remove brackets
            elements = inner_text.split(',')  # split by comma
            text_list = [element.strip().strip('\'"') for element in
                         elements]  # remove leading/trailing spaces and quotes
            print(text_list)
        return text_list

    dataset = {}
    classnames = list(set(texts))  # Remove duplicates and use unique class names
    for class_name in classnames:
        print(f"Starting for class {class_name}...")
        start_time = time.time()  # Start timing

        # prompt = text_prompt(class_name, classnames)
        prompt = text_prompt(class_name, classnames)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            # model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=70,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        res_content = response['choices'][0]['message']['content']
        # print("res=",res_content)
        # dataset[class_name] = clean_responce(res_content)
        dataset[class_name] = res_content
        # print("res=",res_content)

        end_time = time.time()  # End timing
        print(f"Time taken for {class_name}: {end_time - start_time} seconds")

    return dataset


# Example usage in your forward function:
api_base = "xx"
api_key = "sk-MRXtVGX4hnlt4921s2Wixp1zyu9DYUJCm8xKryhGBX4PvMzo"


# nouns = ['dog', 'cat', 'car']  # Example nouns extracted from texts
# def read_txt_file(filename):
#     try:
#         with open(filename, 'r') as file:
#             content = file.read().splitlines()
#             return content
#     except FileNotFoundError:
#         return "File not found"
#     except Exception as e:
#         return f"An error occurred: {str(e)}"
# classnames_file="/root/YOLO-World/LLM/known_classnames.txt"
# classnames = read_txt_file(classnames_file)
def read_json_and_convert(filename):
    # 打开并读取JSON文件
    with open(filename, 'r') as file:
        data = json.load(file)

    # 对每个子列表进行处理，不添加额外引号，让json.dump()处理
    converted_list = [", ".join(sublist) for sublist in data]
    return converted_list


# 使用这个函数读取并转换数据
filename = '/root/YOLO-World/data/texts/lvis_v1_class_texts.json'  # 替换为你的文件名
classnames = read_json_and_convert(filename)
description = generate_descriptions(classnames, api_base, api_key)
print(json.dumps(description, indent=4))
with open('/root/YOLO-World/LLM/unique_description_LVIS1203_GPT4.json', 'w') as file:
    json.dump(description, file, indent=4)
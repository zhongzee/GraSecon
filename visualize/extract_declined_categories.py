import os

def extract_categories_from_file(input_file, output_file):
    categories = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # 跳过空行和标题行
            if not line.strip() or line.startswith('All categories'):
                continue
            # 提取类别名
            parts = line.strip().split()
            if len(parts) >= 1:
                category_name = parts[0]
                # 去掉下划线，替换为空格
                category_name_clean = category_name.replace('_', ' ')
                categories.append(category_name_clean)
    # 将类别名保存到输出文件
    with open(output_file, 'w') as f_out:
        for category in categories:
            f_out.write(category + '\n')
    print(f"已处理文件：{input_file}")
    print(f"提取的类别数量：{len(categories)}")
    print(f"结果已保存到：{output_file}\n")

def process_levels(base_path, levels):
    for level in levels:
        input_file = os.path.join(base_path, level, f"{level}_all_drops_sorted.txt")
        output_file = os.path.join(base_path, level, f"{level}_declined_categories.txt")
        if os.path.exists(input_file):
            extract_categories_from_file(input_file, output_file)
        else:
            print(f"文件不存在：{input_file}\n")

if __name__ == "__main__":
    # 指定基础路径
    base_path = './visualize/inat'
    # 级别列表，从 l6 到 l1
    levels = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']
    # 处理各级别的文件
    process_levels(base_path, levels)

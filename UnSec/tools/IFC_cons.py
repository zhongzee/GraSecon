def validate_parent_child_mappings(level_names, level_hierarchy, gpt_results_root, category_name_to_id):
    """
    验证所有 parent_names 和 child_names 是否在其对应层级中都有映射。
    
    Args:
        level_names (list): 当前项目使用的层级名称列表，如 ['l3', 'l2', 'l1'] 或 ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']。
        level_hierarchy (list): 层级名称按从高到低（或从低到高）排序的列表，用于确定父子层级位置。
        gpt_results_root (str): 存放各层级 GPT 结果 JSON 文件的根目录路径。
        category_name_to_id (defaultdict): 嵌套的类别名称到 ID 的映射结构，形如：
            {
                "l1": {"liquid": "1", "instrument": "2", ...},
                "l2": {...},
                ...
            }
    
    Returns:
        missing_parents (set): 缺少映射的父节点集合。
        missing_children (set): 缺少映射的子节点集合。
    """
    import os
    import json
    from collections import defaultdict

    def load_json(path):
        with open(path, 'r') as f:
            return json.load(f)

    missing_parents = set()
    missing_children = set()

    for level_name in level_names:
        gpt_results_path = os.path.join(gpt_results_root, f"cleaned_{level_name}.json")
        if not os.path.exists(gpt_results_path):
            # 如果某个层级的JSON文件不存在，可以选择直接跳过或处理异常
            continue

        # 读取该层级的 JSON 结果
        gpt_results = load_json(gpt_results_path)

        for entry in gpt_results.values():
            # 获取当前层级的索引
            current_level_index = level_hierarchy.index(level_name)

            # ---------------------
            # 检查 parent_names
            # ---------------------
            for parent_name in entry.get("parent_names", []):
                normalized_parent = parent_name.lower()
                # 假设父节点位于当前层级的上一级
                parent_level_index = current_level_index + 1
                if parent_level_index < len(level_hierarchy):
                    parent_level = level_hierarchy[parent_level_index]
                    # 在对应层级的 category_name_to_id 中查找
                    if normalized_parent not in category_name_to_id[parent_level]:
                        missing_parents.add(f"{parent_level}: {parent_name}")
                else:
                    # 如果已经是最高层级，就不存在更高层级了
                    missing_parents.add(f"No higher level for parent '{parent_name}' in level '{level_name}'")

            # ---------------------
            # 检查 child_names
            # ---------------------
            for child_name in entry.get("child_names", []):
                normalized_child = child_name.lower()
                # 假设子节点位于当前层级的下一级
                child_level_index = current_level_index - 1
                if child_level_index >= 0:
                    child_level = level_hierarchy[child_level_index]
                    # 在对应层级的 category_name_to_id 中查找
                    if normalized_child not in category_name_to_id[child_level]:
                        missing_children.add(f"{child_level}: {child_name}")
                else:
                    # 如果已经是最低层级，就不存在更低层级了
                    missing_children.add(f"No lower level for child '{child_name}' in level '{level_name}'")

    return missing_parents, missing_children
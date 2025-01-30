import argparse
import json
import torch
import numpy as np
import itertools
from nltk.corpus import wordnet
import sys
import clip
import os
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import re
from copy import deepcopy
from collections import defaultdict

import ast
from UnSec.tools.llm_controllers import LLMBot, HrchyPrompter
from UnSec.tools.composer import SignatureComposer
from UnSec.tools.fileios import *

import argparse
import json
import os
import torch
from UnSec.tools.llm_controllers import LLMBot, HrchyPrompter
from UnSec.tools.fileios import dump_json


def compose_isa(signature_list):
    return ['a ' + signature[0] + ''.join([f', which is a {parentName}' for parentName in signature[1:]])
            for signature in signature_list]


def try_deserialize_ast(input_str):
    try:
        return ast.literal_eval(input_str)
    except (ValueError, SyntaxError):
        return [input_str]


def remove_symbols(messy_sent):
    messy_sent = str(messy_sent)
    messy_sent = messy_sent.replace('[', '')
    messy_sent = messy_sent.replace(']', '')
    messy_sent = messy_sent.strip()

    for _ in range(3):
        if messy_sent.startswith("'") or messy_sent.startswith(",") or messy_sent.startswith("."):
            messy_sent = messy_sent[1:].strip()
    for _ in range(3):
        if messy_sent.endswith("'") or messy_sent.endswith(",") or messy_sent.endswith("."):
            messy_sent = messy_sent[:-1].strip()
    return messy_sent.strip()


def organize_sentences(sentences):
    if '\n' in sentences:
        raw_reply = sentences.split('\n')
    else:
        raw_replay = remove_symbols(sentences)
        raw_reply = raw_replay.split("',")

    cleaned_reply = [remove_symbols(raw_sent) for raw_sent in raw_reply]
    return cleaned_reply

def generate_unique_entries(ppt):
    return list(set([entry for entry in ppt]))


def filter_hyperedges(attributes_hyperedge, functional_hyperedge, morphological_hyperedge, ecological_hyperedge):
    # 将所有内容转换为集合去重
    attributes_set = set(attributes_hyperedge.split(' & '))
    functional_set = set(functional_hyperedge.split(' & '))
    morphological_set = set(morphological_hyperedge.split(' & '))
    ecological_set = set(ecological_hyperedge.split(' & '))

    # 过滤重复内容：优先保留到 attributes_set
    functional_set -= attributes_set
    morphological_set -= attributes_set
    ecological_set -= attributes_set

    # 确保 functional 和 morphological 的独特性
    functional_set -= morphological_set
    ecological_set -= functional_set | morphological_set

    # 将集合内容重新转换回字符串格式
    attributes_hyperedge = ' & '.join(attributes_set)
    functional_hyperedge = ' & '.join(functional_set)
    morphological_hyperedge = ' & '.join(morphological_set)
    ecological_hyperedge = ' & '.join(ecological_set)

    return attributes_hyperedge, functional_hyperedge, morphological_hyperedge, ecological_hyperedge


def generate_and_categorize_nodes(prompter, bot, node_name, classification_level):
    # 定义超边描述，用于不同关联关系的查询
    hyperedge_descriptions = [
        ('Distinctive Biological Classification Traits:Highlight broad-level biological traits that define the classification of this group.', classification_level[1]),
        ('Behavioral and Functional Roles:The typical behaviors or roles this organism fulfills in its ecosystem,just answer Noun or noun phrase.', classification_level[1]),
        ('External Structural and Visual Features: The prominent visual features and structural traits that define this organism’s external appearance.just answer Noun or noun phrase.', classification_level[1]),
        ('Ecological Environments: The primary habitats where this organism typically resides, with details on its ecological contexts.just answer Noun or noun phrase.', classification_level[1])
    ]

    # Step 1: 生成与每个超边相连的属性描述
    attributes_hyperedge = bot.infer(
        prompter.embed(node_name, context=[hyperedge_descriptions[0][0], classification_level[1]]))
    functional_hyperedge = bot.infer(
        prompter.embed(node_name, context=[hyperedge_descriptions[1][0], classification_level[1]]))
    morphological_hyperedge = bot.infer(
        prompter.embed(node_name, context=[hyperedge_descriptions[2][0], classification_level[1]]))
    ecological_hyperedge = bot.infer(
        prompter.embed(node_name, context=[hyperedge_descriptions[3][0], classification_level[1]]))

    # 过滤重复项，确保每个超边中的条目是唯一的
    attributes_hyperedge, functional_hyperedge, morphological_hyperedge, ecological_hyperedge = filter_hyperedges(
        attributes_hyperedge, functional_hyperedge, morphological_hyperedge, ecological_hyperedge
    )

    # 返回去重后的超边描述和节点列表
    return attributes_hyperedge, functional_hyperedge, morphological_hyperedge, ecological_hyperedge

def generate_and_categorize_nodes_fsod(prompter, bot, node_name, classification_level):
    # 定义超边描述，用于不同关联关系的查询
    # 更新的 hyperedge_descriptions，用于对象类数据
    hyperedge_descriptions = [
        ('Distinctive Object Classification Traits: Highlight core traits that define the object class or type.',
         classification_level[1]),
        (
        'Functional Roles: Describe the primary functions or uses this object typically fulfills in real-world scenarios. Just answer in nouns or noun phrases.',
        classification_level[1]),
        (
        'External Structural and Visual Features: Describe key visual and structural features that define this object’s appearance. Just answer in nouns or noun phrases.',
        classification_level[1]),
        (
        'Associated Environments: List the common locations or contexts where this object is usually found. Just answer in nouns or noun phrases.',
        classification_level[1])
    ]

    # Step 1: 生成与每个超边相连的属性描述
    attributes_hyperedge = bot.infer(
        prompter._query_fsod(node_name, context=[hyperedge_descriptions[0][0], classification_level[1]]))
    functional_hyperedge = bot.infer(
        prompter._query_fsod(node_name, context=[hyperedge_descriptions[1][0], classification_level[1]]))
    morphological_hyperedge = bot.infer(
        prompter._query_fsod(node_name, context=[hyperedge_descriptions[2][0], classification_level[1]]))
    ecological_hyperedge = bot.infer(
        prompter._query_fsod(node_name, context=[hyperedge_descriptions[3][0], classification_level[1]]))

    # 过滤重复项，确保每个超边中的条目是唯一的
    attributes_hyperedge, functional_hyperedge, morphological_hyperedge, ecological_hyperedge = filter_hyperedges(
        attributes_hyperedge, functional_hyperedge, morphological_hyperedge, ecological_hyperedge
    )

    # 返回去重后的超边描述和节点列表
    return attributes_hyperedge, functional_hyperedge, morphological_hyperedge, ecological_hyperedge

# def generate_and_categorize_nodes(prompter, bot, node_name, child_names, classification_level):
#
#     # hyperedge = ['key attributes', 'Behavioral and Functional Roles', 'Structural and Physiological Features', 'ecological environments']
#     # hyperedge = ['Biological Classification Traits', 'main functions', 'Structural and Physiological Features','ecological environments']
#
#     hyperedge = [
#         'Distinctive Biological Classification Traits: Highlight defining biological characteristics used in taxonomic classifications, with broad-level anatomical or physiological features that differentiate this group. For example: notochord, dorsal nerve cord, pharyngeal slits, post-anal tail.',
#         'Behavioral and Functional Roles: the primary functional roles and behaviors this organism exhibits in its ecosystem that are visually or contextually observable, just demand-generated nouns. For example: underwater respiration, tree-dwelling, migratory movement, pack hunting.',
#         'External Structural and Visual Features: Focus on prominent, visually identifiable structural traits such as body symmetry, segmentation, or distinctive appendages. For example: segmented body plan, paired limbs, streamlined body, coloration patterns.',
#         'Ecological Environments: Specify the main habitats or biomes where this organism typically resides, with more precise placements within these regions. For example: tropical rainforest canopy, freshwater lake edges, coral reef zones, arid desert plains.'
#     ]
#
#     # Step 1: 生成属性描述
#     attributes_hyperedge = bot.infer(prompter.embed(node_name, context=[hyperedge[0], classification_level]))
#     functional_hyperedge = bot.infer(prompter.embed(node_name, context=[hyperedge[1], classification_level]))
#     morphological_hyperedge = bot.infer(prompter.embed(node_name, context=[hyperedge[2], classification_level]))
#     ecological_hyperedge = bot.infer(prompter.embed(node_name, context=[hyperedge[3], classification_level]))
#
#     # 过滤重复项
#     attributes_hyperedge, functional_hyperedge, morphological_hyperedge, ecological_hyperedge = filter_hyperedges(
#         attributes_hyperedge, functional_hyperedge, morphological_hyperedge, ecological_hyperedge
#     )
#
#     # 输出过滤后的结果
#     print(f"Filtered Attributes Hyperedge for {node_name}: {attributes_hyperedge}")
#     print(f"Filtered Functional Hyperedge for {node_name}: {functional_hyperedge}")
#     print(f"Filtered Morphological Hyperedge for {node_name}: {morphological_hyperedge}")
#     print(f"Filtered Ecological Hyperedge for {node_name}: {ecological_hyperedge}")
#
#
#     categorized_nodes = {
#         'attributes_nodes': [],
#         'functional_nodes': [],
#         'morphological_nodes': [],
#         'ecological_nodes': []
#     }
#
#     # Step 2: LLM分类
#     for child_name in child_names:
#         # 构造一个 prompt，让 LLM 判断 child_name 应该在哪一个超边下连接 node_name
#         # prompt = f"Given the context of '{node_name}', classify the entity '{child_name}' into the most appropriate category for association with '{node_name}' based on its properties.\n\n" \
#         #          f"1. Attributes (Biological Classification Traits): {attributes_hyperedge}\n" \
#         #          f"2. Functional (Behavioral and Functional Roles): {functional_hyperedge}\n" \
#         #          f"3. Morphological (External Structural and Visual Features): {morphological_hyperedge}\n" \
#         #          f"4. Ecological (Ecological Environments): {ecological_hyperedge}\n\n" \
#         #          f"Just Answer with 'Attributes', 'Functional', 'Morphological', or 'Ecological' depending on which category best represents the relationship of '{child_name}' to '{node_name}'."
#
#         prompt = f"""
#         Based on the broader biological context of '{node_name}', classify the entity '{child_name}' into the category that best characterizes its relationship to '{node_name}' by its most representative properties. Choose the most suitable association category from the following:
#         1. **Attributes (Biological Classification Traits)**: Traits that highlight biological classifications and essential structural features that define broad categories within biology. Example: {attributes_hyperedge}.
#         2. **Functional (Behavioral and Functional Roles)**: Behaviors or roles this organism typically performs within its ecosystem, highlighting observable actions or ecological roles. Example: {functional_hyperedge}.
#         3. **Morphological (External Structural and Visual Features)**: Visually identifiable structural features, including symmetry, segmentation, and other major body structures that define its external form. Example: {morphological_hyperedge}.
#         4. **Ecological (Ecological Environments)**: Primary habitats and environmental contexts where the organism is typically found, focusing on its usual surroundings or adaptations to specific ecological zones. Example: {ecological_hyperedge}.
#         Only answer with 'Attributes', 'Functional', 'Morphological', or 'Ecological' to best describe the most relevant category for '{child_name}' in its association with '{node_name}'.
#         """.strip()
#
#         category = bot.infer(prompt, temperature=0.7).strip()
#
#         # 将子节点分配到相应类别
#         if category == 'Attributes':
#             categorized_nodes['attributes_nodes'].append(child_name)
#         elif category == 'Functional':
#             categorized_nodes['functional_nodes'].append(child_name)
#         elif category == 'Morphological':
#             categorized_nodes['morphological_nodes'].append(child_name)
#         elif category == 'Ecological':
#             categorized_nodes['ecological_nodes'].append(child_name)
#
#     return attributes_hyperedge, functional_hyperedge, morphological_hyperedge, ecological_hyperedge, categorized_nodes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='fsod', choices=['inat', 'fsod'])
    parser.add_argument('--mode', default='postprocess', choices=['query', 'postprocess'])
    parser.add_argument('--output_root', default='')
    parser.add_argument('--h_level', default='l1', choices=['l6', 'l5', 'l4', 'l3', 'l2', 'l1'])
    parser.add_argument('--num_sub', type=int, default=10)
    parser.add_argument('--num_super', type=int, default=3)
    parser.add_argument('--query_times', type=int, default=1)
    parser.add_argument('--input_file', default='inat_llm_answers',
                        help='Input JSON directory containing hierarchical structure.')

    args = parser.parse_args()
    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    # if not os.path.exists(args.output_root):
    #     raise FileExistsError("Output folder does not exist.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    CHATGPT_ZOO = ['gpt-3.5-turbo'] #  # gpt-3.5-turbo\gpt-3.5-turbo-16k
    bot = LLMBot(CHATGPT_ZOO[0])

    args.output_path = os.path.join(args.output_root, f"raw_{args.dataset_name}_gpt_hrchy_{args.h_level}")

    # Main processing logic
    if args.mode == 'query' and args.dataset_name == 'inat':
        level_names = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']
        level_contexts = {
            'l6': ['types', 'Species'],
            'l5': ['Specie', 'Genus'],
            'l4': ['Genus', 'Family'],
            'l3': ['Family', 'Order'],
            'l2': ['Orders', 'Class'],
            'l1': ['Class', 'Phylum']
        }
        starting_idx = level_names.index(args.h_level)
        level_names = level_names[starting_idx:]

        print('Loading hierarchy tree from inat_annotations/inat_hierarchy_tree.json')
        meta_tree = json.load(open('inat_annotations/inat_hierarchy_tree.json', 'r'))
        meta_level = meta_tree.get(args.h_level)

        if args.h_level == 'l1':
            signature_ids = [[int(x['id'])] for x in sorted(meta_level['categories'], key=lambda x: x['id'])]
            signature_names = [[x['name']] for x in sorted(meta_level['categories'], key=lambda x: x['id'])]

            node_ids = [str(sig_id[0]) for sig_id in signature_ids]
            node_names = [sig_name[0] for sig_name in signature_names]
            node_parents = [[] for sig_name in signature_names]
        else:
            signature_ids = [[int(x['id'])] for x in sorted(meta_level['categories'], key=lambda x: x['id'])]
            for i in range(len(signature_ids)):
                leaf_id = str(signature_ids[i][0])
                parents_ids = meta_level['parents'].get(leaf_id)
                signature_ids[i].extend(parents_ids)

            signature_names = []
            for cat_id in signature_ids:
                cat_name = []
                for level_idx, this_id in enumerate(cat_id):
                    level_name = level_names[level_idx]
                    this_name = meta_tree[level_name]['categories'][this_id - 1]['name']
                    cat_name.append(this_name)
                signature_names.append(cat_name)

            node_ids = [str(sig_id[0]) for sig_id in signature_ids]
            node_names = [sig_name[0] for sig_name in signature_names]
            node_parents = [sig_name[1:] for sig_name in signature_names]

        results_dict = {cat_id: {
            'node_name': cat_name,
            'parent_names': cat_parents,
            'attributes_hyperedge': [],
            'functional_hyperedge': [],
            'morphological_hyperedge': [],
            'ecological_hyperedge': [],
            'child_names': []
        } for cat_id, cat_name, cat_parents in zip(node_ids, node_names, node_parents)}

        # Load the hierarchical structure from input file to get child names
        input_file_path = os.path.join(args.input_file, f'cleaned_inat_gpt_hrchy_{args.h_level}.json')

        if not os.path.exists(input_file_path):
            print(f"Input file {input_file_path} does not exist. Exiting...")
        else:
            with open(input_file_path, 'r') as file:
                hierarchical_structure = json.load(file)

            h_prompter = HrchyPrompter(dataset_name=args.dataset_name, num_sub=args.num_sub, num_super=args.num_super)

            for cat_id, cat_entry in results_dict.items():
                node_name = cat_entry['node_name']
                classification_level = level_contexts[args.h_level]
                child_names = hierarchical_structure.get(cat_id, {}).get('child_names', [])

                print(f"Processed category {cat_id}: {node_name}")

                # Generate and categorize nodes
                attributes_hyperedge, functional_hyperedge, morphological_hyperedge, ecological_hyperedge = \
                    generate_and_categorize_nodes(h_prompter, bot, node_name, classification_level)

                cat_entry['attributes_hyperedge'] = attributes_hyperedge
                cat_entry['functional_hyperedge'] = functional_hyperedge
                cat_entry['morphological_hyperedge'] = morphological_hyperedge
                cat_entry['ecological_hyperedge'] = ecological_hyperedge
                cat_entry['child_names'] = child_names
            # Save results
            # 保存查询结果
            dump_json(args.output_path, results_dict)
            print(f"Detailed descriptions saved to {args.output_path}")

    elif args.mode == 'query' and args.dataset_name == 'fsod':
        level_names = ['l3', 'l2', 'l1']
        level_contexts = {
            'l3': ['types', 'object'],
            'l2': ['types', 'object'],
            'l1': ['types', 'object'],
        }
        starting_idx = level_names.index(args.h_level)
        level_names = level_names[starting_idx:]

        print('Loading hierarchy tree from fsod_annotations/fsod_hierarchy_tree.json')
        meta_tree = json.load(open('fsod_annotations/fsod_hierarchy_tree.json', 'r'))
        meta_level = meta_tree.get(args.h_level)

        if args.h_level == 'l1':
            signature_ids = [[int(x['id'])] for x in sorted(meta_level['categories'], key=lambda x: x['id'])]
            signature_names = [[x['name']] for x in sorted(meta_level['categories'], key=lambda x: x['id'])]

            node_ids = [str(sig_id[0]) for sig_id in signature_ids]
            node_names = [sig_name[0] for sig_name in signature_names]
            node_parents = [[] for sig_name in signature_names]
        else:
            signature_ids = [[int(x['id'])] for x in sorted(meta_level['categories'], key=lambda x: x['id'])]
            for i in range(len(signature_ids)):
                leaf_id = str(signature_ids[i][0])
                parents_ids = meta_level['parents'].get(leaf_id)
                signature_ids[i].extend(parents_ids)

            signature_names = []
            for cat_id in signature_ids:
                cat_name = []
                for level_idx, this_id in enumerate(cat_id):
                    level_name = level_names[level_idx]
                    this_name = meta_tree[level_name]['categories'][this_id - 1]['name']
                    cat_name.append(this_name)
                signature_names.append(cat_name)

            node_ids = [str(sig_id[0]) for sig_id in signature_ids]
            node_names = [sig_name[0] for sig_name in signature_names]
            node_parents = [sig_name[1:] for sig_name in signature_names]

        results_dict = {cat_id: {
            'node_name': cat_name,
            'parent_names': cat_parents,
            'attributes_hyperedge': [],
            'functional_hyperedge': [],
            'morphological_hyperedge': [],
            'ecological_hyperedge': [],
            'child_names': []
        } for cat_id, cat_name, cat_parents in zip(node_ids, node_names, node_parents)}

        # Load the hierarchical structure from input file to get child names
        input_file_path = os.path.join(args.input_file, f'cleaned_inat_gpt_hrchy_{args.h_level}.json')

        if not os.path.exists(input_file_path):
            print(f"Input file {input_file_path} does not exist. Exiting...")
        else:
            with open(input_file_path, 'r') as file:
                hierarchical_structure = json.load(file)

            h_prompter = HrchyPrompter(dataset_name=args.dataset_name, num_sub=args.num_sub, num_super=args.num_super)

            for cat_id, cat_entry in results_dict.items():
                node_name = cat_entry['node_name']
                classification_level = level_contexts[args.h_level]
                child_names = hierarchical_structure.get(cat_id, {}).get('child_names', [])

                print(f"Processed category {cat_id}: {node_name}")

                # Generate and categorize nodes
                attributes_hyperedge, functional_hyperedge, morphological_hyperedge, ecological_hyperedge = \
                    generate_and_categorize_nodes_fsod(h_prompter, bot, node_name, classification_level)

                cat_entry['attributes_hyperedge'] = attributes_hyperedge
                cat_entry['functional_hyperedge'] = functional_hyperedge
                cat_entry['morphological_hyperedge'] = morphological_hyperedge
                cat_entry['ecological_hyperedge'] = ecological_hyperedge
                cat_entry['child_names'] = child_names
            # Save results
            # 保存查询结果
            dump_json(args.output_path, results_dict)
            print(f"Detailed descriptions saved to {args.output_path}")
    elif args.mode == 'query' and args.dataset_name == 'lvis':
        from oid_annotations.class_names import categories_all

        lvis_meta = load_json("lvis_annotations/lvis_v1_train_cat_info.json")
        lvis_cnames = [entry["name"].replace("_", " ").replace("-", " ").lower() for entry in lvis_meta]
        combined_cnames = lvis_cnames # 1203個類別名
        # 该数据集的层次结构信息已经预先通过语言模型生成
        # oid_lvis_results = load_json("openset_lvis_oid_llm_answers/cleaned_oid_lvis_gpt_hrchy_l1.json")

        oid_lvis_results = load_json("lvis_llm_answers/cleaned_lvis_gpt_hrchy_l1.json")

        print(f"{len(combined_cnames)} (combined_cnames) = {len(lvis_cnames)} (lvis)")

        context = ['types', 'object']

        results_dict = {str(1+cat_id): {'node_name': cat_name, 'parent_names': [], 'child_names': [],
                                        'candidate_sentences': []}
                        for cat_id, cat_name in enumerate(combined_cnames)}


        def retrieve_oid_lvis(node_name):
            for key, value in list(oid_lvis_results.items()):
                if value['node_name'] == node_name:
                    return deepcopy(value)

        for cat_id, cat_entry in list(results_dict.items()):
            results_dict[cat_id] = retrieve_oid_lvis(cat_entry['node_name'])

        dump_json(args.output_path, results_dict)
    elif args.mode == 'query' and args.dataset_name == 'coco':
        coco_meta = load_json("coco_annotations/instances_val2017_all_2_oriorder_cat_info.json")
        coco_cnames = [entry["name"].replace("_", " ").replace("-", " ").lower() for entry in coco_meta]
        combined_cnames = coco_cnames

        print(f"{len(combined_cnames)} (combined_cnames) = {len(coco_cnames)} (coco)")

        context = ['types', 'object']

        results_dict = {str(1+cat_id): {'node_name': cat_name, 'parent_names': [], 'child_names': []}
                        for cat_id, cat_name in enumerate(combined_cnames)}

        h_prompter = HrchyPrompter(dataset_name=args.dataset_name, num_sub=args.num_sub, num_super=args.num_super)
        for cat_id, cat_entry in results_dict.items():
            ppt_childs, ppt_parents = h_prompter.embed(node_name=cat_entry['node_name'], context=context)

            child_answers = [bot.infer(ppt_childs, temperature=0.7) for i in range(args.query_times)]
            parent_answers = [bot.infer(ppt_parents, temperature=0.7) for i in range(args.query_times)]

            results_dict[cat_id]['child_names'] = child_answers
            results_dict[cat_id]['parent_names'] = parent_answers

            print(f"[{cat_id}] Question A: {ppt_childs}")
            for i in range(args.query_times):
                print(f"Answer A-{1 + i}: {child_answers[i]}")

            print(f"[{cat_id}] Question B: {ppt_parents}")
            for i in range(args.query_times):
                print(f"Answer B-{1 + i}: {parent_answers[i]}")
            print('\n')
            # if int(cat_id) >= 3:
            #     break

        dump_json(args.output_path, results_dict)

    # # 构建图结构
    # G = build_graph_from_results(results_dict)
    elif args.mode == 'postprocess' and args.dataset_name == 'fsod':
        raw_results = load_json(args.output_path)
        # 新定义一个 composer，用于生成超边句子
        hypergraph_composer = SignatureComposer(prompter='semantic_hypergraph')

        for k_cat, v in raw_results.items():
            print("processd",v["node_name"])
            # 清理子节点
            clean_childs = {
                name.strip()
                for dirty_child in v["child_names"]
                for name in dirty_child.split('&')
                if 3 <= len(name.strip()) <= 100
            }
            v["child_names"] = list(clean_childs)

            # 保留第一个父节点
            parent_names = [v["parent_names"][0]] if v["parent_names"] else []

            # 构造层级信息
            hierarchy_signature_names = [
                [child_name, v["node_name"]] + parent_names
                for child_name in v["child_names"]
            ]

            # 清理其他字段
            def clean_list(dirty_data):
                # 如果 dirty_data 是字符串，直接用 '&' 分割
                if isinstance(dirty_data, str):
                    items = dirty_data.split('&')
                # 如果 dirty_data 是列表，先转换为单个字符串再分割
                elif isinstance(dirty_data, list):
                    items = '&'.join(dirty_data).split('&')
                else:
                    return dirty_data  # 如果是其他数据类型，直接返回

                # 去除空格、过滤无效长度并去重
                clean_items = {item.strip() for item in items if 3 <= len(item.strip()) <= 100}

                # 返回清理后的列表
                return list(clean_items) if clean_items else []


            # 确保每个超边都有数据，即使为空也设置为 []
            v["attributes_hyperedge"] = clean_list(v.get("attributes_hyperedge"))
            v["functional_hyperedge"] = clean_list(v.get("functional_hyperedge"))
            v["morphological_hyperedge"] = clean_list(v.get("morphological_hyperedge"))
            v["ecological_hyperedge"] = clean_list(v.get("ecological_hyperedge"))
            # v["hierarchy_names"] = hierarchy_signature_names

            # 使用清理后的数据生成候选句子
            signature_data = {
                'node_name': v["node_name"],
                'attributes_hyperedge': v["attributes_hyperedge"],
                'functional_hyperedge': v["functional_hyperedge"],
                'morphological_hyperedge': v["morphological_hyperedge"],
                'ecological_hyperedge': v["ecological_hyperedge"],
                # 'hierarchy_names': hierarchy_signature_names  # 将层级信息传递给 composer
            }
            # 确保每个超边的内容都有数据，否则提供空字符串以防止错误
            v["hypergraph_sentences"] = hypergraph_composer.compose([signature_data])

        # 将清理后的数据保存到新文件
        dump_json(args.output_path.replace('raw', 'cleaned'), raw_results)

    elif args.mode == 'postprocess' and args.dataset_name == 'fsod':
        raw_results = load_json(args.output_path)
        isa_composer = SignatureComposer(prompter='isa')

        for k_cat, v in raw_results.items():
            clean_childs = {
                name.strip()
                for dirty_child in v["child_names"]
                for name in dirty_child.split('&')
                if 3 <= len(name.strip()) <= 100
            }

            clean_parents = {
                name.strip()
                for dirty_parent in v["parent_names"]
                for name in dirty_parent.split('&')
                if 3 <= len(name.strip()) <= 100
            }
            v["child_names"] = list(clean_childs)
            v["parent_names"] = list(clean_parents)

            signature_names = [
                [child_name, v["node_name"], parent_name]
                for parent_name in v["parent_names"][:1]
                for child_name in v["child_names"]
            ]

            v["candidate_sentences"] = isa_composer.compose(signature_names)

        dump_json(args.output_path.replace('raw', 'cleaned_'), raw_results)

    elif args.mode == 'postprocess' and args.dataset_name in ['oid', 'coco']:
        raw_results = load_json(args.output_path)
        isa_composer = SignatureComposer(prompter='isa')

        current_cnames = [entry["node_name"].replace("_", " ").replace("-", " ").lower()
                          for _, entry in raw_results.items()]

        for k_cat, v in raw_results.items():
            clean_childs = {
                name.strip()
                for dirty_child in v["child_names"]
                for name in dirty_child.split('&')
                if 3 <= len(name.strip()) <= 100
            }

            clean_parents = {
                name.strip()
                for dirty_parent in v["parent_names"]
                for name in dirty_parent.split('&')
                if 3 <= len(name.strip()) <= 100
            }

            clean_childs = list(clean_childs)
            clean_parents = list(clean_parents)

            trimmed_childs = [name.lower() for name in clean_childs]
            trimmed_parents = [name.lower() for name in clean_parents]

            trimmed_childs = list(set(trimmed_childs))
            trimmed_parents = list(set(trimmed_parents))

            v["child_names"] = clean_childs
            v["parent_names"] = clean_parents

            signature_names = [
                [child_name, v["node_name"], parent_name]
                for parent_name in v["parent_names"]
                for child_name in v["child_names"]
            ]

            v["candidate_sentences"] = isa_composer.compose(signature_names)

        dump_json(args.output_path.replace('raw', 'cleaned'), raw_results)
    elif args.mode == 'postprocess' and args.dataset_name in ['lvis']: # 選這個
        raw_results = load_json(args.output_path.replace('raw', 'cleaned'))
        isa_composer = SignatureComposer(prompter='isa')

        current_cnames = [entry["node_name"].replace("_", " ").replace("-", " ").lower()
                          for _, entry in raw_results.items()]

        for k_cat, v in raw_results.items():
            clean_childs = v["child_names"]
            clean_parents = v["parent_names"]

            trimmed_childs = [name.lower() for name in clean_childs]
            trimmed_parents = [name.lower() for name in clean_parents]

            trimmed_childs = list(set(trimmed_childs))
            trimmed_parents = list(set(trimmed_parents))

            v["child_names"] = clean_childs
            v["parent_names"] = clean_parents

            signature_names = [
                [child_name, v["node_name"], parent_name]
                for parent_name in v["parent_names"]
                for child_name in v["child_names"]
            ]

            v["candidate_sentences"] = isa_composer.compose(signature_names)

        dump_json(args.output_path.replace('raw', 'cleaned'), raw_results)
    else:
        raise NotImplementedError

"""
--mode
query
--dataset_name
lvis
--output_root
lvis_llm_answers_2
"""
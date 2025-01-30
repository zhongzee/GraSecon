import argparse
import torch
import numpy as np
import clip
from collections import defaultdict, OrderedDict
from UnSec.tools.themer import Themer
from UnSec.tools.fileios import *
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
"""
--dataset_name "inat"
--gpt_results_root "inat_llm_answers"
--prompter isa
--aggregator mean
--clip_model "$clip_model"
--out_path "${nexus_paths[$clip_model]}"

--dataset_name
"inat"
--gpt_results_root
"inat_llm_detailed_answers"
--prompter
"isa"
--aggregator
"mean"
--clip_model
"ViT-B/32"
--out_path
".././nexus/lvis/UnSec_llm_detail"

--dataset_name
"fsod"
--gpt_results_root
"fsod_llm_detail_answers"
--prompter
"isa"
--aggregator
"mean"
--clip_model
"ViT-B/32"
--out_path
".././nexus/fsod/vitB32/UnSec_llm_detail"
"""

def select_sentences_by_level(candidate_sentences, hypergraph_sentences, detail_sentences,level_name, sentence_type="by_level"):
    """
    根据层级或模式选择合适的句子列表。

    参数：
    - candidate_sentences (list): 粗粒度句子列表
    - hypergraph_sentences (list): 细粒度句子列表
    - level_name (str): 当前层级名称，用于控制选择
    - sentence_type (str): 使用的句子类型，可选 "by_level", "candidate", "detail", "combined"
        - "by_level"：根据层级智能选择句子
        - "candidate"：只使用粗粒度候选句子
        - "detail"：只使用细粒度详细句子
        - "combined"：直接合并候选句子和详细句子

    返回：
    - list: 选定的句子列表
    """
    if sentence_type == "hierarchy":
        return candidate_sentences
    elif sentence_type == "detail":
        return hypergraph_sentences
    elif sentence_type == "combined":
        return candidate_sentences + hypergraph_sentences
    elif sentence_type == "all_combined":
        return candidate_sentences + hypergraph_sentences + detail_sentences
    elif sentence_type == "by_level":
        # 根据层级选择性地使用句子
        if level_name in ['l1', 'l2']:  # 粗粒度层级
            return candidate_sentences + hypergraph_sentences
        elif level_name in ['l3','l4','l5', 'l6']:  # 细粒度层级
            return hypergraph_sentences
        else:  # 中等层级，结合使用
            return candidate_sentences
    elif sentence_type == "all_combined_by_level":
        # 根据层级选择性地使用句子
        if level_name in ['l1', 'l2']:  # 粗粒度层级
            return candidate_sentences + hypergraph_sentences + detail_sentences
        elif level_name in ['l3','l4','l5', 'l6']:  # 细粒度层级
            return hypergraph_sentences +detail_sentences
        else:  # 中等层级，结合使用
            return candidate_sentences
    else:
        raise ValueError("Unsupported sentence_type. Choose 'by_level', 'candidate', 'detail', or 'combined'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='fsod', choices=['inat', 'fsod'])
    parser.add_argument('--gpt_results_root', default='inat_llm_detail_answers_attributes')
    parser.add_argument('--prompter', default='isa', choices=['a', 'avg', 'concat', 'isa'])
    parser.add_argument('--aggregator', default='mean', choices=['peigen', 'mean', 'mixed', 'all_eigens'])
    parser.add_argument('--peigen_thresh', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--out_path', default='')
    parser.add_argument('--clip_model', default="RN50", choices=['ViT-B/32', 'RN50'])
    parser.add_argument('--sentence_type', default='detail', choices=['by_level', 'candidate', 'detail', 'combined','all_combined','all_combined_by_level'])

    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    if not is_valid_folder(args.out_path): raise FileExistsError

    # Device Selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset_name == 'inat':
        level_names = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']
        # level_names = ['l3', 'l2', 'l1']
    else:
        level_names = ['l3', 'l2', 'l1']

    print('Loading CLIP')
    global_encoder, global_preprocess = clip.load(args.clip_model, device=device)
    theme_maker = Themer(method=args.aggregator, thresh=args.peigen_thresh, alpha=args.alpha)

    theme_tree_features = defaultdict(dict)
    for level_name in level_names:
        gpt_results = load_json(os.path.join(args.gpt_results_root,
                                             f"cleaned_{args.dataset_name}_gpt_detail_hrchy_{level_name}.json"))
        for cat_id, entry in gpt_results.items():
            candidate_sentences = entry["candidate_sentences"]
            hypergraph_sentences = entry["hypergraph_sentences"] # inat 使用inat_llm_detailed_answers
            detail_sentences = entry["detail_sentences"]
            # 合并候选句子和详细句子，并截断
            print("使用",args.sentence_type)
            sentences_to_use = select_sentences_by_level(candidate_sentences, hypergraph_sentences, detail_sentences,level_name,sentence_type=args.sentence_type)
            truncated_sentences = [sentence[:77] for sentence in sentences_to_use]
            node_tokens = clip.tokenize(truncated_sentences).to(device)
            with torch.no_grad():
                node_features = global_encoder.encode_text(node_tokens)
            # node_features = F.normalize(node_features)
            node_theme = theme_maker.get_theme(node_features)
            theme_tree_features[level_name][cat_id] = node_theme

    for level_name, level_ids in theme_tree_features.items():
        total_num = len(list(level_ids.values()))
        print(f"Total feats = {total_num} at {level_name}")

    # Prepare and Save Features
    for level_name, level_theme_dict in theme_tree_features.items():
        sorted_theme_dict = OrderedDict(sorted(level_theme_dict.items(), key=lambda x: int(x[0])))

        l_feats = list(sorted_theme_dict.values())
        l_classifier = torch.stack(l_feats)
        print(f"---> {level_name}'s classifier has a shape of {l_classifier.shape}")

        # Save the embeddings
        path_save = os.path.join(args.out_path, f"{args.dataset_name}_clip_hrchy_{level_name}.npy")
        # print(f'Saving to {path_save}')
        # torch.save(l_classifier.cpu(), path_save)
        print(f'Saving to {path_save}')
        np.save(open(path_save, 'wb'), l_classifier.cpu().numpy())





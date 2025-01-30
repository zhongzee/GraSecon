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
python -W ignore build_inat_fsod_aggr_w_llm_hrchy.py \
       --dataset_name "inat" \
       --gpt_results_root "inat_llm_graph_answers" \
       --prompter isa \
       --aggregator weighted \
       --clip_model "ViT-B/32" \
       --out_path ".././nexus/inat/vitB32/UnSec_llm_graph" \
       --use_child_similarity
   
python -W ignore build_inat_fsod_aggr_w_llm_hrchy.py \    
--dataset_name "fsod" \
--gpt_results_root "fsod_llm_graph_answers" \
--prompter isa \
--aggregator weighted \
--clip_model "ViT-B/32" \
--out_path ".././nexus/fsod/vitB32/UnSec_llm_graph" \
--use_child_similarity

"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='fsod', choices=['inat', 'fsod'])
    parser.add_argument('--gpt_results_root', default='fsod_llm_graph_answers')
    parser.add_argument('--prompter', default='isa', choices=['a', 'avg', 'concat', 'isa'])
    parser.add_argument('--aggregator', default='mean', choices=['peigen', 'mean', 'mixed', 'all_eigens', 'weighted'])
    parser.add_argument('--peigen_thresh', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--out_path', default='')
    parser.add_argument('--clip_model', default="RN50", choices=['ViT-B/32', 'RN50'])
    parser.add_argument('--use_child_similarity', action='store_true', help='Whether to use child similarities as weights for weighted aggregation')

    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    if not is_valid_folder(args.out_path):
        raise FileExistsError

    # Device Selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset_name == 'inat':
        level_names = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']
    else:
        level_names = ['l3', 'l2', 'l1']

    print('Loading CLIP')
    global_encoder, global_preprocess = clip.load(args.clip_model, device=device)
    theme_maker = Themer(method=args.aggregator, thresh=args.peigen_thresh, alpha=args.alpha)

    theme_tree_features = defaultdict(dict)

    for level_name in level_names:
        gpt_results = load_json(os.path.join(args.gpt_results_root, f"cleaned_{args.dataset_name}_gpt_graph_hrchy_{level_name}.json"))
        for cat_id, entry in gpt_results.items():
            node_sentences = entry["candidate_sentences"]
            node_tokens = clip.tokenize(node_sentences).to(device)

            with torch.no_grad():
                node_features = global_encoder.encode_text(node_tokens)

            # 默认使用均值聚合
            weights = None

            # 如果启用了使用子节点相似度作为权重
            if args.use_child_similarity and "child_similarities_with_scores" in entry:
                # 从 gpt_results 中读取子节点的相似度得分作为权重
                weights = [score for _, score in entry["child_similarities_with_scores"]]

            # 获取节点特征（根据聚合方式进行加权或者均值计算）
            node_theme = theme_maker.get_theme(node_features, weights=weights) # torch.Size([1024])
            theme_tree_features[level_name][cat_id] = node_theme

    # 输出节点信息
    for level_name, level_ids in theme_tree_features.items():
        total_num = len(list(level_ids.values()))
        print(f"Total feats = {total_num} at {level_name}") # Total feats = 500 at l6,Total feats = 317 at l5,Total feats = 184 at l4,Total feats = 61 at l3,Total feats = 18 at l2,Total feats = 5 at l1

    # 准备和保存特征
    for level_name, level_theme_dict in theme_tree_features.items():
        sorted_theme_dict = OrderedDict(sorted(level_theme_dict.items(), key=lambda x: int(x[0])))

        l_feats = list(sorted_theme_dict.values())
        l_classifier = torch.stack(l_feats)
        print(f"---> {level_name}'s classifier has a shape of {l_classifier.shape}") # l6's classifier has a shape of torch.Size([500, 1024])

        # 保存嵌入特征
        path_save = os.path.join(args.out_path, f"{args.dataset_name}_clip_hrchy_{level_name}.npy")
        print(f'Saving to {path_save}')
        np.save(open(path_save, 'wb'), l_classifier.cpu().numpy())

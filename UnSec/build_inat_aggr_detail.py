import argparse
import torch
import numpy as np
import clip
from collections import defaultdict, OrderedDict
from copy import deepcopy
from tools.composer import SignatureComposer
from tools.themer import Themer
from tools.fileios import *


def load_detail_sentences(hierarchy_dir, level_names):
    """
    Load detail_sentences for each level from JSON files into a dictionary.
    """
    detail_dict = {}
    for level in level_names:
        file_path = os.path.join(hierarchy_dir, f"cleaned_inat_gpt_detail_hrchy_{level}.json")
        with open(file_path, 'r') as f:
            data = json.load(f)
            for node_id, content in data.items():
                detail_dict[content['node_name']] = content['detail_sentences']
    return detail_dict

def get_mean_vector(stacked_feats):
    print("========== used Mean when shape={}".format(stacked_feats.shape))
    mean_theme = torch.mean(stacked_feats, dim=0)  # mean vector
    return mean_theme

def compute_leaf_embedding(htree, leaf_level, composer, clip_model='ViT-B/32', detail_dict=None, device='cuda'):
    l_names = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']

    meta_level_leaf = htree.get(leaf_level)

    # Extract class ids with their parents
    signature_ids = [[int(x['id'])] for x in sorted(meta_level_leaf['categories'], key=lambda x: x['id'])]
    for i in range(len(signature_ids)):
        leaf_id = str(signature_ids[i][0])
        parents_ids = meta_level_leaf['parents'].get(leaf_id)
        signature_ids[i].extend(parents_ids)

    signature_names = []
    for cat_id in signature_ids:
        cat_name = []
        for level_idx, this_id in enumerate(cat_id):
            level_name = l_names[level_idx]
            this_name = htree[level_name]['categories'][this_id - 1]['name']
            cat_name.append(this_name)
        signature_names.append(cat_name)

    assert len(signature_ids) == len(signature_names)
    assert all(len(signature_id) == len(signature_name) for signature_id, signature_name in
               zip(signature_ids, signature_names))

    # Compose sentences from signatures
    sentences = composer.compose(signature_names)

    for sent in sentences:
        print("orin_sent",sent)

    # Match and append detail sentences to each signature
    detail_sentences = []
    for signature in signature_names:
        details = []
        for name in signature:
            if name in detail_dict:
                details.extend(detail_dict[name])
        detail_sentences.append(details)

    for sent in detail_sentences:
        print("detai_sent",sent)

    print('Loading CLIP')
    model, preprocess = clip.load(clip_model, device=device)

    # Encode sentences
    text = clip.tokenize(sentences).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)

    # Encode detail sentences and combine
    detail_features = []

    # Encode detail_sentences and calculate mean
    for detail_set in detail_sentences:
        if not detail_set:
            continue
        detail_tokens = clip.tokenize(detail_set).to(device)
        with torch.no_grad():
            encoded_details = model.encode_text(detail_tokens)
        avg_detail_feature = get_mean_vector(encoded_details)  # Use _get_mean_vector
        detail_features.append(avg_detail_feature)

    # Ensure detail features match text features length
    assert len(detail_features) == len(text_features), "Mismatch between sentences and detail_sentences encoding."

    # Combine features
    combined_features = []
    for text_feat, detail_feat in zip(text_features, detail_features):
        stacked_feats = torch.stack([text_feat, detail_feat])
        combined = get_mean_vector(stacked_feats)  # Use _get_mean_vector
        combined_features.append(combined)

    # Return combined features
    return torch.stack(combined_features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', default='./UnSec/inat_annotations/inat_hierarchy_tree.json')
    parser.add_argument('--prompter', default='isa', choices=['a', 'concat', 'isa'])
    parser.add_argument('--aggregator', default='mean', choices=['peigen', 'mean', 'mixed', 'all_eigens'])
    parser.add_argument('--out_path', default='./nexus/inat/vitB32/UnSec_inat_detail')
    parser.add_argument('--peigen_thresh', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--clip_model', default="ViT-B/32", choices=['ViT-B/32', 'RN50'])

    args = parser.parse_args()

    import os

    if not os.path.exists(args.out_path):
        print(f"Output directory '{args.out_path}' does not exist. Creating it...")
        os.makedirs(args.out_path)


    if not is_valid_folder(args.out_path):
        raise FileExistsError

    # Device Selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print('Loading CLIP')
    global_encoder, global_preprocess = clip.load(args.clip_model, device=device)

    # Load Metadata
    print(f'Loading {args.tree}')
    level_names = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']

    meta_tree = json.load(open(args.tree, 'r'))
    meta_level_leaf = meta_tree.get('l6')

    # Load detail sentences
    hierarchy_dir = "./UnSec/inat_llm_detail_answers/"
    detail_dict = load_detail_sentences(hierarchy_dir, level_names)

    # Extract class ids with their parents
    signature_ids = [[int(x['id'])] for x in sorted(meta_level_leaf['categories'], key=lambda x: x['id'])]
    for i in range(len(signature_ids)):
        leaf_id = str(signature_ids[i][0])
        parents_ids = meta_level_leaf['parents'].get(leaf_id)
        signature_ids[i].extend(parents_ids)

    tree_childs_to_leaf = {
        'l1': {},
        'l2': {},
        'l3': {},
        'l4': {},
        'l5': {},
    }

    for leaf_signature in signature_ids:
        cat_id_at_leaf = leaf_signature[0]
        for level_idx, level_name in enumerate(level_names[1:], start=1):
            this_level_parent_id = str(leaf_signature[level_idx])
            if this_level_parent_id in tree_childs_to_leaf[level_name]:
                tree_childs_to_leaf[level_name][this_level_parent_id].append(cat_id_at_leaf)
            else:
                tree_childs_to_leaf[level_name][this_level_parent_id] = [cat_id_at_leaf]

    prompt_composer = SignatureComposer(prompter=args.prompter)

    # Compute Leaf Features
    leaf_features = compute_leaf_embedding(
        meta_tree, level_names[0], composer=prompt_composer,
        clip_model=args.clip_model, detail_dict=detail_dict,
        device=device
    )

    theme_maker = Themer(method=args.aggregator, thresh=args.peigen_thresh, alpha=args.alpha)

    print(leaf_features.shape)

    tree_features = defaultdict(dict)
    for i, feat in enumerate(leaf_features, start=1):
        tree_features[level_names[0]][str(i)] = feat

    for level_name, level_childs in tree_childs_to_leaf.items():
        for level_cat_id, level_child_ids in level_childs.items():
            print(f"{level_name} cat_{level_cat_id} has childs: {level_child_ids}")

            level_child_feats = [tree_features[level_names[0]][str(idx)]
                                 for idx in level_child_ids]
            tree_features[level_name][level_cat_id] = level_child_feats

    theme_tree_features = deepcopy(tree_features)

    for level_name, level_ids in tree_features.items():
        if level_name == 'l6':
            continue

        for level_cat_id, level_child_feats in level_ids.items():
            stacked_child_feats = torch.stack(level_child_feats)
            theme_feat = theme_maker.get_theme(stacked_child_feats)
            theme_tree_features[level_name][level_cat_id] = theme_feat

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
        path_save = os.path.join(args.out_path, f"inat_clip_hrchy_{level_name}.npy")
        print(f'Saving to {path_save}')
        np.save(open(path_save, 'wb'), l_classifier.cpu().numpy())

import argparse
import torch
import numpy as np
import sys
import clip
from UnSec.tools.composer import SignatureComposer
from UnSec.tools.fileios import *

"""
--prompter isa
--clip_model "RN50"
--out_path ".././nexus/inat/rn50/isa"

--prompter isa
--clip_model "ViT-B/32"
--out_path ".././nexus/inat/vitB32/isa"
                      
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', default='inat_annotations/inat_hierarchy_tree.json')
    parser.add_argument('--prompter', default='a', choices=['a', 'avg', 'concat', 'isa'])
    # parser.add_argument('--level', default='l6', choices=['l6', 'l5', 'l4', 'l3', 'l2', 'l1'])
    parser.add_argument('--out_path', default='')
    parser.add_argument('--clip_model', default="ViT-B/32", choices=['ViT-B/32', 'RN50'])

    args = parser.parse_args()

    if not is_valid_folder(args.out_path): raise FileExistsError
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for this_tree_level in ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']:
        args.level = this_tree_level
        path_save = os.path.join(args.out_path, f"inat_clip_hrchy_{args.level}.npy")

        print('Loading', args.tree)

        # load metadata
        level_names = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']
        starting_idx = level_names.index(args.level)
        level_names = level_names[starting_idx:]

        meta_tree = json.load(open(args.tree, 'r'))
        meta_level = meta_tree.get(args.level)
        if args.prompter == 'a' or args.level == 'l1':
            print('Loading CLIP')
            model, preprocess = clip.load(args.clip_model, device=device)

            sentences = ["a {}".format(x['name']) for x in sorted(meta_level['categories'], key=lambda x: x['id'])]

            print(sentences)
            text_features = []
            for sig_list in sentences:
                # tokenize class names
                sig_text = clip.tokenize(sig_list).to(device)

                # encoding
                with torch.no_grad():
                    sig_text_features = model.encode_text(sig_text)
                    # print(sig_text_features.shape)

                text_features.append(sig_text_features.mean(dim=0))
            text_features = torch.stack(text_features, dim=0)

            print('text_features.shape', text_features.shape)
            text_features = text_features.cpu().numpy()

            print('saveing to', path_save)
            np.save(open(path_save, 'wb'), text_features)
        else:
            prompt_composer = SignatureComposer(prompter=args.prompter)
            # extract class ids w/ its parents
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
                    this_name = meta_tree[level_name]['categories'][this_id-1]['name']
                    cat_name.append(this_name)
                signature_names.append(cat_name)

            assert len(signature_ids) == len(signature_names)
            assert all(len(signature_id) == len(signature_name) for signature_id, signature_name in
                       zip(signature_ids, signature_names))

            sentences = prompt_composer.compose(signature_names)

            print('Loading CLIP')
            model, preprocess = clip.load(args.clip_model, device=device)

            if args.prompter in ['concat', 'isa', 'a']:
                # tokenize class names
                text = clip.tokenize(sentences).to(device)

                # encoding
                with torch.no_grad():
                    if len(text) > 10000:
                        text_features = torch.cat([
                            model.encode_text(text[:len(text) // 2]),
                            model.encode_text(text[len(text) // 2:])],
                            dim=0)
                    else:
                        text_features = model.encode_text(text)

                print('text_features.shape', text_features.shape)
                text_features = text_features.cpu().numpy()

                print('saveing to', path_save)
                np.save(open(path_save, 'wb'), text_features)
            elif args.prompter in ['avg']:
                text_features = []
                for sig_list in sentences:
                    # tokenize class names
                    sig_text = clip.tokenize(sig_list).to(device)

                    # encoding
                    with torch.no_grad():
                        sig_text_features = model.encode_text(sig_text)
                        # print(sig_text_features.shape)

                    text_features.append(sig_text_features.mean(dim=0))
                text_features = torch.stack(text_features, dim=0)

                print('text_features.shape', text_features.shape)
                text_features = text_features.cpu().numpy()

                print('saveing to', path_save)
                np.save(open(path_save, 'wb'), text_features)
            else:
                raise NotImplementedError



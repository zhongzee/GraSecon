#!/bin/bash

conda activate GraSecon

cd GraSecon_cls || exit

echo "Planting COCO LLM hierarchy tree to: GraSecon/coco_llm_answers"

python -W ignore plant_llm_syn_hrchy_tree.py \
                  --mode query \
                  --dataset_name coco \
                  --output_root coco_llm_answers

python -W ignore plant_llm_syn_hrchy_tree.py \
                  --mode postprocess \
                  --dataset_name coco \
                  --output_root coco_llm_answers
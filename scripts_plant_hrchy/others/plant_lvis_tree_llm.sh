#!/bin/bash

conda activate GraSecon

cd GraSecon_cls || exit

echo "Planting LVIS LLM hierarchy tree to: GraSecon/lvis_llm_answers"

python -W ignore plant_llm_syn_hrchy_tree.py \
                  --mode query \
                  --dataset_name lvis \
                  --output_root lvis_llm_answers

python -W ignore plant_llm_syn_hrchy_tree.py \
                  --mode postprocess \
                  --dataset_name lvis \
                  --output_root lvis_llm_answers
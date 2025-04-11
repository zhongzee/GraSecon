#!/bin/bash

conda activate GraSecon

cd GraSecon_cls || exit

echo "Planting ImageNet-1k LLM-generated hierarchy ground-truth hierarchy tree to: GraSecon_cls/imagenet1k"

python -W ignore plant_hierarchy.py --source "chatgpt"

python -W ignore plant_hierarchy.py --source "chatgpt_post"
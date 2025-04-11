#!/bin/bash

conda activate GraSecon

cd GraSecon_cls || exit

echo "Planting FSOD ground-truth hierarchy tree to: GraSecon/fsod_annotations/fsod_hierarchy_tree.json"

python -W ignore plant_fsod_hrchy_tree.py
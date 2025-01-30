#!/bin/bash

conda activate UnSec

cd UnSec_cls || exit

echo "Planting FSOD ground-truth hierarchy tree to: UnSec/fsod_annotations/fsod_hierarchy_tree.json"

python -W ignore plant_fsod_hrchy_tree.py
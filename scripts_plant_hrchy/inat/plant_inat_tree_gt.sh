#!/bin/bash

conda activate UnSec

cd UnSec_cls || exit

echo "Planting iNat ground-truth hierarchy tree to: UnSec/inat_annotations/inat_hierarchy_tree.json"

python -W ignore plant_inat_hrchy_tree.py
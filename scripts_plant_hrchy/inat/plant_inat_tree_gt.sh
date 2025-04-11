#!/bin/bash

conda activate GraSecon

cd GraSecon_cls || exit

echo "Planting iNat ground-truth hierarchy tree to: GraSecon/inat_annotations/inat_hierarchy_tree.json"

python -W ignore plant_inat_hrchy_tree.py
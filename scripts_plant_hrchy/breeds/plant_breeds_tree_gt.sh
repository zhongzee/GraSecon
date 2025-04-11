#!/bin/bash

conda activate GraSecon

cd GraSecon_cls || exit

echo "Planting BREEDS hierarchy ground-truth hierarchy tree to: GraSecon_cls/hrchy_breeds"

python -W ignore plant_hierarchy.py --source "breeds"
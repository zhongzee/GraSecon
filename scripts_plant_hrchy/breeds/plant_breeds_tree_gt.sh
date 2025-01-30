#!/bin/bash

conda activate UnSec

cd UnSec_cls || exit

echo "Planting BREEDS hierarchy ground-truth hierarchy tree to: UnSec_cls/hrchy_breeds"

python -W ignore plant_hierarchy.py --source "breeds"
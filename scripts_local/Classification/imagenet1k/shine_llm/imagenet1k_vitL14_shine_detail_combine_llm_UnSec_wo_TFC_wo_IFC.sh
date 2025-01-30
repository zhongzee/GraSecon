#!/bin/bash

activateAndRun() {
    # Activate the conda environment
    conda activate UnSec

    # Change to the specified directory, exit if it fails
    cd UnSec_cls || exit

   # If you wanna test inference speed, change --num_runs to 10
    python -W ignore zeroshot_UnSec_wo_TFC_wo_IFC.py \
              --model_size "ViT-L/14" \
              --method "UnSec" \
              --hierarchy_tree_path "./UnSec_cls/hrchy_imagenet1k/imagenet1k_detail_llm_composed.json" \
              --batch_size 1 \
              --num_runs 10
}

# Call the function
activateAndRun


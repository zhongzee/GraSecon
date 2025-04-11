#!/bin/bash

activateAndRun() {
    # Activate the conda environment
    conda activate GraSecon

    # Change to the specified directory, exit if it fails
    cd GraSecon_cls || exit

   # If you wanna test inference speed, change --num_runs to 10
    python -W ignore zeroshot_GraSecon_wo_TFC_wo_IFC.py \
              --model_size "ViT-L/14" \
              --method "GraSecon" \
              --hierarchy_tree_path "./GraSecon_cls/hrchy_imagenet1k/imagenet1k_detail_llm_composed.json" \
              --batch_size 1 \
              --num_runs 10
}

# Call the function
activateAndRun


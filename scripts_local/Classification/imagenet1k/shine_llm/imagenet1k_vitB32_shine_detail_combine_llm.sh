#!/bin/bash

activateAndRun() {
    # Activate the conda environment
    conda activate GraSecon

    # Change to the specified directory, exit if it fails
    cd GraSecon_cls || exit

   # If you wanna test inference speed, change --num_runs to 10
    python -W ignore zeroshot.py \
              --model_size "ViT-B/32" \
              --method "GraSecon" \
              --hierarchy_tree_path "hrchy_imagenet1k/imagenet1k_detail_llm_composed.json" \
              --batch_size 1 \
              --num_runs 10
}

# Call the function
activateAndRun


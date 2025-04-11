#!/bin/bash

activateAndRun() {
    # Activate the conda environment
    conda activate GraSecon

    # Change to the specified directory, exit if it fails
    cd GraSecon_cls || exit

   # If you wanna test inference speed, change --num_runs to 10
    python -W ignore zeroshot.py \
              --model_size "ViT-B/16" \
              --method "GraSecon" \
              --hierarchy_tree_path "imagenet1k_hrchy_wordnet.json" \
              --batch_size 64 \
              --num_runs 1
}

# Call the function
activateAndRun


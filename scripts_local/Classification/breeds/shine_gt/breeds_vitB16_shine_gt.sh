#!/bin/bash

activateAndRun() {
    # Activate the conda environment
    conda activate UnSec

    # Change to the specified directory, exit if it fails
    cd UnSec_cls || exit

    # Define breed levels in an array
    local breed_levels=(l6 l5 l4 l3 l2 l1)

    # Loop through each breed level
    for level in "${breed_levels[@]}"; do
        python -W ignore zeroshot_breeds.py \
                  --model_size "ViT-B/16" \
                  --method "UnSec" \
                  --breed_level "$level" \
                  --batch_size 64
    done
}

# Call the function
activateAndRun

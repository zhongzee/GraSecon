#!/bin/bash

activateEnvironmentAndMove() {
    # Activate the conda environment
    conda activate GraSecon

    # Change directory
    cd GraSecon || exit
}

buildNexus() {
    declare -A nexus_paths=(
        ["ViT-B/32"]=".././nexus/inat/vitB32/GraSecon_gt"
        ["RN50"]=".././nexus/inat/rn50/GraSecon_gt"
    )

    for clip_model in "${!nexus_paths[@]}"; do
        python -W ignore build_inat_aggr.py \
                          --prompter isa \
                          --aggregator mean \
                          --clip_model "$clip_model" \
                          --out_path "${nexus_paths[$clip_model]}"
    done
}

# Main script execution
activateEnvironmentAndMove
buildNexus

#!/bin/bash

activateEnvironmentAndMove() {
    # Activate the conda environment
    conda activate GraSecon

    # Change directory
    cd GraSecon || exit
}

buildNexus() {
    declare -A nexus_paths=(
        ["ViT-B/32"]=".././nexus/fsod/vitB32/isa"
        ["RN50"]=".././nexus/fsod/rn50/isa"
    )

    for clip_model in "${!nexus_paths[@]}"; do
        python -W ignore build_fsod_sing.py \
                          --prompter isa \
                          --clip_model "$clip_model" \
                          --out_path "${nexus_paths[$clip_model]}"
    done
}

# Main script execution
activateEnvironmentAndMove
buildNexus

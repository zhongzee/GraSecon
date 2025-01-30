#!/bin/bash

activateEnvironmentAndMove() {
    # Activate the conda environment
    conda activate UnSec

    # Change directory
    cd UnSec || exit
}

buildNexus() {
    declare -A nexus_paths=(
        ["ViT-B/32"]=".././nexus/fsod/vitB32/UnSec_gt"
        ["RN50"]=".././nexus/fsod/rn50/UnSec_gt"
    )

    for clip_model in "${!nexus_paths[@]}"; do
        python -W ignore build_fsod_aggr.py \
                          --prompter isa \
                          --aggregator mean \
                          --clip_model "$clip_model" \
                          --out_path "${nexus_paths[$clip_model]}"
    done
}

# Main script execution
activateEnvironmentAndMove
buildNexus

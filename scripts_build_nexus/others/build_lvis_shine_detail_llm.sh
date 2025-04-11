#!/bin/bash

activateEnvironmentAndMove() {
    # Activate the conda environment
    conda activate GraSecon

    # Change directory
    cd GraSecon || exit
}

buildNexus() {
    declare -A nexus_paths=(
        ["ViT-B/32"]=".././nexus/lvis/GraSecon_detail_llm_combine"
    )

    for clip_model in "${!nexus_paths[@]}"; do
        python -W ignore build_miss_inat_fsod_aggr_w_llm_detail_hrchy.py \
                          --dataset_name "lvis" \
                          --prompter isa \
                          --aggregator mean \
                          --clip_model "$clip_model" \
                          --out_path "${nexus_paths[$clip_model]}"
    done
}

# Main script execution
activateEnvironmentAndMove
buildNexus

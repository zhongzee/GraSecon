#!/bin/bash

activateEnvironmentAndMove() {
    # Activate the conda environment
    conda activate GraSecon

    # Change directory
    cd GraSecon || exit
}

buildNexus() {
    declare -A nexus_paths=(
        ["ViT-B/32"]=".././nexus/fsod/vitB32/GraSecon_llm"
        ["RN50"]=".././nexus/fsod/rn50/GraSecon_llm"
    )

    for clip_model in "${!nexus_paths[@]}"; do
        python -W ignore build_inat_fsod_aggr_w_llm_hrchy.py \
                          --dataset_name "fsod" \
                          --gpt_results_root "fsod_llm_answers" \
                          --prompter isa \
                          --aggregator mean \
                          --clip_model "$clip_model" \
                          --out_path "${nexus_paths[$clip_model]}"
    done
}

# Main script execution
activateEnvironmentAndMove
buildNexus

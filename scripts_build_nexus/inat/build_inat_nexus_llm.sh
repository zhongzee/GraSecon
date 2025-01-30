#!/bin/bash

activateEnvironmentAndMove() {
    # Activate the conda environment
    conda activate UnSec

    # Change directory
    cd UnSec || exit
}

buildNexus() {
    declare -A nexus_paths=(
        ["ViT-B/32"]=".././nexus/inat/vitB32/UnSec_llm"
        ["RN50"]=".././nexus/inat/rn50/UnSec_llm"
    )

    for clip_model in "${!nexus_paths[@]}"; do
        python -W ignore build_inat_fsod_aggr_w_llm_hrchy.py \
                          --dataset_name "inat" \
                          --gpt_results_root "inat_llm_answers" \
                          --prompter isa \
                          --aggregator mean \
                          --clip_model "$clip_model" \
                          --out_path "${nexus_paths[$clip_model]}"
    done
}

# Main script execution
activateEnvironmentAndMove
buildNexus

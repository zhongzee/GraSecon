#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mem=64000
#SBATCH --time 15-00:00:00
#SBATCH --output=./slurm-output/cls_imagenet1k_vitB16_GraSecon_llm.out

export PATH="/home/mliu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

activateAndRun() {
    # Activate the conda environment
    conda activate GraSecon

    # Change to the specified directory, exit if it fails
    cd GraSecon_cls || exit

   # If you wanna test inference speed, change --num_runs to 10
    python -W ignore zeroshot.py \
              --model_size "ViT-B/16" \
              --method "GraSecon" \
              --hierarchy_tree_path "imagenet1k_hrchy_llm_composed.json" \
              --batch_size 256 \
              --num_runs 1
}

# Call the function
activateAndRun


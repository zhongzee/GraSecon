#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mem=64000
#SBATCH --time 15-00:00:00
#SBATCH --output=./slurm-output/cls_imagenet1k_vitB16_baseline.out

export PATH="/home/mliu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

activateAndRun() {
    # Activate the conda environment
    conda activate UnSec

    # Change to the specified directory, exit if it fails
    cd UnSec_cls || exit

   # If you wanna test inference speed, change --num_runs to 10
    python -W ignore zeroshot.py \
              --model_size "ViT-B/16" \
              --method "zeroshot" \
              --batch_size 256 \
              --num_runs 1
}

# Call the function
activateAndRun

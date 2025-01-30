#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:8
#SBATCH --mem=64000
#SBATCH --time 15-00:00:00
#SBATCH --output=./slurm-output/inat_miss_detic_SwinB_LVIS-IN-21K_UnSec_llm.out

export PATH="/home/mliu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate UnSec

METADATA_ROOT="./nexus/inat_miss/vitB32/UnSec_llm"

python train_net_detic.py \
        --num-gpus 8 \
        --config-file ./configs_detic/only_test_Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
        --eval-only \
        DATASETS.TEST "('inat_val_l6',)" \
        MODEL.WEIGHTS ./models/detic/cross_eval/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('${METADATA_ROOT}/inat_clip_hrchy_l6.npy', )" \
        MODEL.TEST_NUM_CLASSES "(1966,)" \
        MODEL.MASK_ON False

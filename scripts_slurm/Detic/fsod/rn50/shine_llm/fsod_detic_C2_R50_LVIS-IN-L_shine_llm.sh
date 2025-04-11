#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:8
#SBATCH --mem=64000
#SBATCH --time 15-00:00:00
#SBATCH --output=./slurm-output/fsod_detic_C2_R50_LVIS-IN-L_GraSecon_llm.out

export PATH="/home/mliu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate GraSecon

METADATA_ROOT="./nexus/fsod/vitB32/GraSecon_llm"
python train_net_detic.py \
        --num-gpus 8 \
        --config-file ./configs_detic/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size.yaml\
        --eval-only \
        DATASETS.TEST "('fsod_test_l1', 'fsod_test_l2', 'fsod_test_l3',)" \
        MODEL.WEIGHTS ./models/detic/lvis_std/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size.pth \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "(${METADATA_ROOT}/fsod_clip_hrchy_l1.npy', ${METADATA_ROOT}/fsod_clip_hrchy_l2.npy', ${METADATA_ROOT}/fsod_clip_hrchy_l3.npy',)" \
        MODEL.TEST_NUM_CLASSES "(15, 46, 200,)" \
        MODEL.MASK_ON False

#!/bin/bash

conda activate GraSecon

METADATA_ROOT="./nexus/inat_miss/vitB32/GraSecon_llm_detail_GraSecon_epoch1"

python train_net_detic_IFC.py \
        --num-gpus 8 \
        --config-file ./configs_detic/only_test_Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
        --eval-only \
        DATASETS.TEST "('inat_val_l6',)" \
        MODEL.WEIGHTS ./models/detic/cross_eval/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('${METADATA_ROOT}/inat_clip_hrchy_l6.npy', )" \
        MODEL.TEST_NUM_CLASSES "(1966,)" \
        MODEL.MASK_ON False

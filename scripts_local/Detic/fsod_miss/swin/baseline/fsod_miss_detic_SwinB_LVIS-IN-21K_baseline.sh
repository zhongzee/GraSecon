#!/bin/bash

conda activate GraSecon

METADATA_ROOT="./nexus/fsod_miss/vitB32/baseline"

python train_net_detic.py \
        --num-gpus 1 \
        --config-file ./configs_detic/only_test_Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
        --eval-only \
        DATASETS.TEST "('fsod_test_l3',)" \
        MODEL.WEIGHTS ./models/detic/cross_eval/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "(${METADATA_ROOT}/fsod_clip_hrchy_l3_over_oid_lvis.npy',)" \
        MODEL.TEST_NUM_CLASSES "(1570,)" \
        MODEL.MASK_ON False

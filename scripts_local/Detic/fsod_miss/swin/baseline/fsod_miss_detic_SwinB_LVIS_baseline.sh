#!/bin/bash

conda activate GraSecon

METADATA_ROOT="./nexus/fsod_miss/vitB32/baseline"

python train_net_detic.py \
        --num-gpus 1 \
        --config-file ./configs_detic/BoxSup-C2_L_CLIP_SwinB_896b32_4x.yaml\
        --eval-only \
        DATASETS.TEST "('fsod_test_l3',)" \
        MODEL.WEIGHTS ./models/detic/cross_eval/BoxSup-C2_L_CLIP_SwinB_896b32_4x.pth \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "(${METADATA_ROOT}/fsod_clip_hrchy_l3_over_oid_lvis.npy',)" \
        MODEL.TEST_NUM_CLASSES "(1570,)" \
        MODEL.MASK_ON False

#!/bin/bash

conda activate GraSecon

METADATA_ROOT="./nexus/fsod/vitB32/GraSecon_llm_TFC_nggm_rHSB_1231"

python train_net_detic.py \
        --num-gpus 8 \
        --config-file ./configs_detic/BoxSup-C2_L_CLIP_SwinB_896b32_4x.yaml\
        --eval-only \
        DATASETS.TEST "('fsod_test_l1', 'fsod_test_l2', 'fsod_test_l3',)" \
        MODEL.WEIGHTS ./models/detic/cross_eval/BoxSup-C2_L_CLIP_SwinB_896b32_4x.pth \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('${METADATA_ROOT}/fsod_clip_hrchy_l1.npy', '${METADATA_ROOT}/fsod_clip_hrchy_l2.npy', '${METADATA_ROOT}/fsod_clip_hrchy_l3.npy',)" \
        MODEL.TEST_NUM_CLASSES "(15, 46, 200,)" \
        MODEL.MASK_ON False

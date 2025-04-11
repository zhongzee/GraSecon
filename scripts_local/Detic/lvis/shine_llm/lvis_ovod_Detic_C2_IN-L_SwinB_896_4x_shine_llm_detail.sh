#!/bin/bash

conda activate GraSecon

METADATA_ROOT="./nexus/lvis/GraSecon_llm_detail"

python train_net_detic.py \
        --num-gpus 1 \
        --config-file ./configs_detic/Detic_LbaseI_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
        --eval-only \
        DATASETS.TEST "('lvis_v1_val',)" \
        MODEL.WEIGHTS ./models/detic/lvis_ovod/Detic_LbaseI_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('${METADATA_ROOT}/lvis_clip_hrchy_l1.npy',)" \
        MODEL.TEST_NUM_CLASSES "(1203,)" \
        MODEL.MASK_ON False

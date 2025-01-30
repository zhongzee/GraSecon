#!/bin/bash

conda activate UnSec

METADATA_ROOT="./nexus/inat/vitB32/UnSec_llm_detail_attributes"

python train_net_detic.py \
        --num-gpus 1 \
        --config-file ./configs_detic/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size.yaml\
        --eval-only \
        DATASETS.TEST "('inat_val_l1',)" \
        MODEL.WEIGHTS ./models/detic/lvis_std/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size.pth \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('${METADATA_ROOT}/inat_clip_hrchy_l1.npy', )" \
        MODEL.TEST_NUM_CLASSES "(5,)" \
        MODEL.MASK_ON False

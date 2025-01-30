#!/bin/bash

conda activate UnSec

METADATA_ROOT="./nexus/coco/vitB32/UnSec_llm_detail_unsec"

python train_net_detic_coco.py \
        --num-gpus 8 \
        --config-file ./configs_detic/coco/Detic_OVCOCO_CLIP_R50_1x_max-size.yaml\
        --eval-only \
        MODEL.WEIGHTS ./models/detic/coco_ovod/Detic_OVCOCO_CLIP_R50_1x_max-size.pth \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('${METADATA_ROOT}/coco_clip_hrchy_l1.npy',)" \
        MODEL.TEST_NUM_CLASSES "(80,)" \
        MODEL.MASK_ON False
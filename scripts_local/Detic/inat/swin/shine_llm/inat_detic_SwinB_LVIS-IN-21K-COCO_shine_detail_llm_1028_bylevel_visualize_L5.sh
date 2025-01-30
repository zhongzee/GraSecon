#!/bin/bash

conda activate UnSec

METADATA_ROOT="./nexus/inat/vitB32/inat_llm_detail_answers_1028"

python train_net_detic_visuallize.py \
        --num-gpus 1 \
        --config-file ./configs_detic/only_test_Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
        --eval-only \
        DATASETS.TEST "('inat_val_l5',)" \
        MODEL.WEIGHTS ./models/detic/cross_eval/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('${METADATA_ROOT}/inat_clip_hrchy_l5.npy', )" \
        MODEL.TEST_NUM_CLASSES "(317,)" \
        MODEL.MASK_ON False

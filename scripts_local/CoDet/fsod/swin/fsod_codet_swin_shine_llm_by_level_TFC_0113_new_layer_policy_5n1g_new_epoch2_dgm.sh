#!/bin/bash

conda activate GraSecon

CFG_SWIN="./configs_codet/CoDet_OVLVIS_SwinB_4x_ft4x.yaml"
MODEL_SWIN="./models/codet/CoDet_OVLVIS_SwinB_4x_ft4x.pth"

METADATA_ROOT="./nexus/fsod/vitB32/GraSecon_by_level_TFC_nggm_rHSB_1231_w_new_TFC_w_SR_epoch2"

python train_net_codet_IFC.py \
        --num-gpus 8 \
        --config-file ${CFG_SWIN} \
        --eval-only \
        DATASETS.TEST "('fsod_test_l1', 'fsod_test_l2', 'fsod_test_l3',)" \
        MODEL.WEIGHTS ${MODEL_SWIN} \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('${METADATA_ROOT}/fsod_clip_hrchy_l1.npy', '${METADATA_ROOT}/fsod_clip_hrchy_l2.npy', '${METADATA_ROOT}/fsod_clip_hrchy_l3.npy',)" \
        MODEL.TEST_NUM_CLASSES "(15, 46, 200,)" \
        MODEL.MASK_ON False


#!/bin/bash

conda activate UnSec

# Configuration files
CFG_VL_SWIN="configs_vldet/VLDet_LbaseI_CLIP_SwinB_896b32_2x_ft4x_caption.yaml"
# Model weight files
W_VL_SWIN="models/vldet/lvis_vldet_swinB.pth"

METADATA_ROOT="./nexus/fsod/rn50/UnSec_by_level_TFC_nggm_rHSB_1231_w_new_TFC_w_SR_epoch2"

python train_net_vldet_IFC.py \
        --num-gpus 8 \
        --config-file ${CFG_VL_SWIN} \
        --eval-only \
        DATASETS.TEST "('fsod_test_l1', 'fsod_test_l2', 'fsod_test_l3',)" \
        MODEL.WEIGHTS ${W_VL_SWIN} \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('${METADATA_ROOT}/fsod_clip_hrchy_l1.npy', '${METADATA_ROOT}/fsod_clip_hrchy_l2.npy', '${METADATA_ROOT}/fsod_clip_hrchy_l3.npy',)" \
        MODEL.TEST_NUM_CLASSES "(15, 46, 200,)" \
        MODEL.MASK_ON False

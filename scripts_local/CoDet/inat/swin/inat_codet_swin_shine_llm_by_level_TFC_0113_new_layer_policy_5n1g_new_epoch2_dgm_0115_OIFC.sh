#!/bin/bash

conda activate GraSecon

CFG_SWIN="./configs_codet/CoDet_OVLVIS_SwinB_4x_ft4x.yaml"
MODEL_SWIN="./models/codet/CoDet_OVLVIS_SwinB_4x_ft4x.pth"

METADATA_ROOT="./nexus/inat/vitB32/GraSecon_llm_by_level_TFC_0113_new_layer_policy_5n1g_new_epoch2_dgm"

python train_net_codet_IFC_OIFC.py \
        --num-gpus 8 \
        --config-file ${CFG_SWIN} \
        --eval-only \
        DATASETS.TEST "('inat_val_l1', 'inat_val_l2', 'inat_val_l3','inat_val_l4','inat_val_l5','inat_val_l6',)" \
        MODEL.WEIGHTS ${MODEL_SWIN} \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('${METADATA_ROOT}/inat_clip_hrchy_l1.npy', '${METADATA_ROOT}/inat_clip_hrchy_l2.npy', '${METADATA_ROOT}/inat_clip_hrchy_l3.npy', '${METADATA_ROOT}/inat_clip_hrchy_l4.npy', '${METADATA_ROOT}/inat_clip_hrchy_l5.npy', '${METADATA_ROOT}/inat_clip_hrchy_l6.npy', )" \
        MODEL.TEST_NUM_CLASSES "(5, 18, 61, 184, 317, 500,)" \
        MODEL.MASK_ON False


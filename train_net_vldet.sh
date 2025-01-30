#!/bin/bash

# SCRIPT_PATH="./scripts_local/Detic/inat/swin/UnSec_gt_detail_domain/inat_detic_SwinB_LVIS-IN-21K-COCO_UnSec_gt_detail_TFC_inat_policy_dgm_0113.sh"

# SCRIPT_PATH="./scripts_local/CoDet/inat/rn50/inat_codet_rn50_UnSec_llm_by_level_TFC_0113_new_layer_policy_5n1g_new_epoch2_dgm_IFC.sh"

# SCRIPT_PATH="./scripts_local/CoDet/inat/swin/inat_codet_swin_UnSec_llm_by_level_TFC_0113_new_layer_policy_5n1g_new_epoch2_dgm.sh"

# SCRIPT_PATH="./scripts_local/VLDet/inat/swin/inat_VLDet_Swin_llm_by_level_TFC_0113_new_layer_policy_5n1g_new_epoch2_dgm.sh"

# SCRIPT_PATH="./scripts_local/VLDet/inat/rn50/inat_VLDet_R50_UnSec_by_level_TFC_0113_new_layer_policy_5n1g_new_epoch2_dgm.sh"

SCRIPT_PATH="./scripts_local/VLDet/inat/swin/inat_VLDet_Swin_llm_by_level_TFC_0113_new_layer_policy_5n1g_new_epoch2_dgm.sh"

SCRIPT_PATH="./scripts_local/VLDet/inat/swin/inat_VLDet_Swin_llm_by_level_TFC_0113_new_layer_policy_5n1g_new_epoch2_dgm_swin.sh"

# SCRIPT_PATH="./scripts_local/VLDet/inat/swin/inat_VLDet_Swin_llm_by_level_TFC_0113_new_layer_policy_5n1g_new_epoch2_dgm_swin.sh"
SCRIPT_PATH="./scripts_local/VLDet/inat/swin/inat_VLDet_Swin_llm_by_level_TFC_0113_new_layer_policy_5n1g_new_epoch2_dgm_swin_OIFC.sh" # 这里必须改zero-shot=512

# SCRIPT_PATH="./scripts_local/VLDet/fsod/swin/fsod_VLDet_Swin_UnSec_llm_level_TFC_0113_new_layer_policy_5n1g_new_epoch2_dgm.sh"

SCRIPT_PATH="./scripts_local/VLDet/fsod/rn50/fsod_VLDet_R50_UnSec_llm_level_TFC_0113_new_layer_policy_5n1g_new_epoch2_dgm.sh"

SCRIPT_PATH="./scripts_local/VLDet/fsod/swin/fsod_VLDet_Swin_UnSec_llm.sh"

SCRIPT_PATH="./scripts_local/VLDet/fsod/rn50/fsod_VLDet_R50_UnSec_llm_by_level_TFC_nggm_rHSB_1231.sh"
# 设置日志的基础路径
BASE_LOG_DIR="./log"

# 获取脚本的相对路径
RELATIVE_PATH=$(realpath --relative-to="./UnSec-master" "$SCRIPT_PATH")

# 去掉脚本文件名的扩展名（.sh），用于日志文件名
LOG_FILE_NAME=$(basename "$RELATIVE_PATH" .sh)

# 构建完整的日志文件路径
LOG_FILE="${BASE_LOG_DIR}/${RELATIVE_PATH%.sh}.log"

# 确保日志文件的目录存在
LOG_DIR=$(dirname "$LOG_FILE")
mkdir -p "$LOG_DIR"

# 使用nohup执行脚本，并将输出重定向到日志文件（覆盖旧文件）
nohup bash "$SCRIPT_PATH" > "$LOG_FILE" 2>&1 &

# 提示日志文件路径
echo "Script is running in background, log file: $LOG_FILE"

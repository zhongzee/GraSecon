#!/bin/bash

# 脚本的绝对路径
#SCRIPT_PATH="./scripts_local/Detic/inat/swin/UnSec_llm/inat_detic_SwinB_LVIS-IN-21K_UnSec_detail_llm.sh"
#SCRIPT_PATH="./scripts_local/Detic/coco/UnSec_llm/coco_ovod_Detic_CLIP_Caption-image_R50_1x_UnSec_llm_detail.sh"

#SCRIPT_PATH="./scripts_local/Detic/lvis/UnSec_llm/lvis_ovod_Detic_C2_CCimg_R50_640_4x_UnSec_llm_detail.sh"

#SCRIPT_PATH="./scripts_local/Detic/inat/rn50/UnSec_llm/inat_detic_C2_R50_LVIS-IN-L_UnSec_detail_llm.sh"

#SCRIPT_PATH="./scripts_local/Detic/inat/swin/UnSec_llm/inat_detic_SwinB_LVIS-IN-21K-COCO_UnSec_combined_llm.sh"

#SCRIPT_PATH="./scripts_local/Detic/fsod/rn50/UnSec_llm/fsod_detic_C2_R50_LVIS-IN-L_UnSec_detail_llm_combine.sh"

#SCRIPT_PATH="./scripts_local/Detic/inat/swin/UnSec_llm_graph/inat_detic_SwinB_LVIS-IN-21K-COCO_UnSec_graph_all_combined_bylevel_llm.sh"

#SCRIPT_PATH="./scripts_local/Detic/fsod/swin/UnSec_llm/fsod_detic_SwinB_LVIS-IN-21K_UnSec_detail_llm.sh"

#SCRIPT_PATH="./scripts_local/Detic/fsod/swin/UnSec_llm/fsod_detic_SwinB_LVIS-IN-21K-COCO_UnSec_detail_combined_llm.sh"
#SCRIPT_PATH="./scripts_local/Detic/inat/rn50/UnSec_llm/inat_detic_C2_R50_LVIS-IN-L_UnSec_detail_llm-by-level.sh"
#SCRIPT_PATH="./scripts_local/Detic/fsod/swin/UnSec_llm_graph/fsod_detic_SwinB_LVIS-IN-21K-COCO_UnSec_detail_graph_bylevel_llm.sh"

SCRIPT_PATH="./scripts_local/Classification/imagenet1k/UnSec_llm/imagenet1k_vitL14_UnSec_detail_combine_llm.sh"
# 获取脚本所在的目录
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

# 获取脚本的基本名称，用于创建日志文件
SCRIPT_NAME=$(basename "$SCRIPT_PATH" .sh)

# 构建日志文件的完整路径
LOG_FILE="${SCRIPT_DIR}/${SCRIPT_NAME}.log"

# 确保日志文件的目录存在
mkdir -p "$SCRIPT_DIR"

# 使用nohup执行脚本，并将输出重定向到日志文件（覆盖旧文件）
nohup bash "$SCRIPT_PATH" > "$LOG_FILE" 2>&1 &

echo "Script is running in background, log file: $LOG_FILE"

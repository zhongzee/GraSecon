#!/bin/bash

# 脚本的绝对路径 FSOD
# SCRIPT_PATH="./scripts_local/Detic/fsod/swin/GraSecon_domain/fsod_detic_SwinB_LVIS-IN-21K-COCO_GraSecon_detail_llm_MFC_TFC.sh"

# SCRIPT_PATH="./scripts_local/Detic/fsod/swin/GraSecon_domain/fsod_detic_SwinB_LVIS-IN-21K-COCO_GraSecon_llm_MFC_TFC.sh"

# SCRIPT_PATH="./scripts_local/Detic/fsod/swin/GraSecon_domain/fsod_detic_SwinB_LVIS-IN-21K-COCO_GraSecon_llm_MFC_TFC.sh"

# SCRIPT_PATH="./scripts_local/Detic/fsod/swin/GraSecon_domain/fsod_detic_SwinB_LVIS-IN-21K-COCO_GraSecon_by_level_llm_TFC.sh"

# SCRIPT_PATH="./scripts_local/Detic/fsod/swin/GraSecon_domain/fsod_detic_SwinB_LVIS-IN-21K-COCO_GraSecon_by_level_llm_MFC_TFC.sh"

# iNat
# SCRIPT_PATH="./scripts_local/Detic/inat/swin/GraSecon_llm_domain/inat_detic_SwinB_LVIS-IN-21K-COCO_GraSecon_by_level_llm_1218_TFC.sh"

# SCRIPT_PATH="./scripts_local/Detic/inat/swin/GraSecon_llm_domain/inat_detic_SwinB_LVIS-IN-21K-COCO_GraSecon_llm_1218_IFC_TFC.sh"

# SCRIPT_PATH="./scripts_local/Detic/inat/swin/GraSecon_llm_domain/inat_detic_SwinB_LVIS-IN-21K-COCO_GraSecon_llm_1218_TFC.sh"

SCRIPT_PATH="./scripts_local/Detic/inat/swin/GraSecon_llm_domain/inat_detic_SwinB_LVIS-IN-21K-COCO_GraSecon_llm_1218_IFC_TFC_L4_L6.sh"

# 设置日志的基础路径
BASE_LOG_DIR="./log"

# 获取脚本的相对路径
RELATIVE_PATH=$(realpath --relative-to="./GraSecon-master" "$SCRIPT_PATH")

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

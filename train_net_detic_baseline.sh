#!/bin/bash

# 脚本的绝对路径
SCRIPT_PATH="/root/UnSec/scripts_local/Detic/inat/swin/baseline/inat_detic_SwinB_LVIS-IN-21K-COCO_baseline.sh"

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

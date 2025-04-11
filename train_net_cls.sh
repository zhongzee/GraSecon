#!/bin/bash

# SCRIPT_PATH="./scripts_local/Classification/imagenet1k/GraSecon_llm/imagenet1k_vitB16_GraSecon_detail_combine_llm_GraSecon.sh"

# SCRIPT_PATH="./scripts_local/Classification/imagenet1k/GraSecon_llm/imagenet1k_vitB32_GraSecon_detail_combine_llm_GraSecon.sh"

# SCRIPT_PATH="./scripts_local/Classification/imagenet1k/GraSecon_llm/imagenet1k_vitL14_GraSecon_detail_combine_llm_GraSecon.sh"

# SCRIPT_PATH="./scripts_local/Classification/imagenet1k/GraSecon_llm/imagenet1k_vitL14_GraSecon_detail_combine_llm_GraSecon_wo_TFC.sh"

# SCRIPT_PATH="./scripts_local/Classification/imagenet1k/GraSecon_llm/imagenet1k_vitB32_GraSecon_detail_combine_llm_GraSecon_wo_TFC.sh"

SCRIPT_PATH="./scripts_local/Classification/imagenet1k/GraSecon_llm/imagenet1k_vitB16_GraSecon_detail_combine_llm_GraSecon_wo_TFC.sh"

SCRIPT_PATH="./scripts_local/Classification/imagenet1k/GraSecon_llm/imagenet1k_vitB16_GraSecon_detail_combine_llm_GraSecon_wo_TFC_wo_IFC.sh"

SCRIPT_PATH="./scripts_local/Classification/imagenet1k/GraSecon_llm/imagenet1k_vitB32_GraSecon_detail_combine_llm_GraSecon_wo_TFC_wo_IFC.sh"

SCRIPT_PATH="./scripts_local/Classification/imagenet1k/GraSecon_llm/imagenet1k_vitL14_GraSecon_detail_combine_llm_GraSecon_wo_TFC_wo_IFC.sh"

SCRIPT_PATH="./scripts_local/Classification/imagenet1k/GraSecon_llm/imagenet1k_vitB16_GraSecon_detail_combine_llm_GraSecon_wo_TFC_1runs.sh"

SCRIPT_PATH="./scripts_local/Classification/imagenet1k/GraSecon_llm/imagenet1k_vitB32_GraSecon_detail_combine_llm_GraSecon_wo_TFC_wo_IFC_1runs.sh"

SCRIPT_PATH="./scripts_local/Classification/imagenet1k/GraSecon_llm/imagenet1k_vitL14_GraSecon_detail_combine_llm_GraSecon_wo_TFC_wo_IFC_1runs.sh"

SCRIPT_PATH="./scripts_local/Classification/imagenet1k/GraSecon_llm/imagenet1k_vitB16_GraSecon_detail_combine_llm_GraSecon_wo_TFC_wo_IFC_1runs.sh"
# SCRIPT_PATH="./scripts_local/Classification/imagenet1k/GraSecon_llm/imagenet1k_vitB16_GraSecon_detail_combine_llm_GraSecon_wo_TFC_wo_IFC_1runs.sh"
# lvis

# SCRIPT_PATH="./scripts_local/Detic/lvis/GraSecon_llm/lvis_ovod_Detic_C2_IN-L_SwinB_896_4x_GraSecon_llm_detail_GraSecon.sh"

# SCRIPT_PATH="./scripts_local/Detic/lvis/GraSecon_llm/lvis_ovod_Detic_C2_IN-L_R50_640_4x_GraSecon_llm_detail_GraSecon.sh"

# SCRIPT_PATH="./scripts_local/Detic/lvis/GraSecon_llm/lvis_ovod_Detic_C2_CCimg_R50_640_4x_GraSecon_llm_detail_GraSecon.sh"

# SCRIPT_PATH="./scripts_local/Detic/lvis/GraSecon_llm/lvis_ovod_BoxSup_C2_Lbase_CLIP_R50_640_4x_GraSecon_detail_GraSecon.sh"

# coco

# SCRIPT_PATH="/mn./log/scripts_local/Detic/inat_misst/afs/huangtao3/wzz/GraSecon-master/scripts_local/Detic/coco/GraSecon_llm/coco_ovod_Detic_CLIP_image_R50_1x_GraSecon_llm_GraSecon_woIFC.sh"

# SCRIPT_PATH="./scripts_local/Detic/coco/GraSecon_llm/coco_ovod_Detic_CLIP_Caption-image_R50_1x_GraSecon_llm_detail_GraSecon_wo_IFC.sh"


# SCRIPT_PATH="./scripts_local/Detic/coco/GraSecon_llm/coco_ovod_Detic_CLIP_caption_R50_1x_GraSecon_llm_GraSecon_wo_IFC.sh"

# SCRIPT_PATH="./scripts_local/Detic/coco/GraSecon_llm/coco_ovod_BoxSup_CLIP_R50_1x_GraSecon_llm_detail_GraSecon_wo_IFC.sh"

# SCRIPT_PATH="./scripts_local/Detic/coco/GraSecon_llm/coco_ovod_BoxSup_CLIP_R50_1x_GraSecon_llm_detail_GraSecon.sh"

# SCRIPT_PATH="./scripts_local/Detic/coco/GraSecon_llm/coco_ovod_Detic_CLIP_caption_R50_1x_GraSecon_llm_GraSecon.sh"

# SCRIPT_PATH="./scripts_local/Classification/breeds/GraSecon_gt/breeds_vitB16_GraSecon_gt.sh"
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

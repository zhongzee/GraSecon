#!/bin/bash

# 目标 torch 版本
TARGET_VERSION="2.4.1"

echo "Checking torch versions for all Conda environments..."
echo "Target torch version: $TARGET_VERSION"
echo "---------------------------------------------"

# 遍历所有 Conda 环境
for env in $(conda env list | awk '{print $1}' | grep -v '^#'); do
    # 激活环境并检查 torch 版本
    TORCH_VERSION=$(conda run -n $env python -c "import torch; print(torch.__version__)" 2>/dev/null)
    if [ $? -eq 0 ]; then
        # 打印每个环境的 torch 版本
        echo "Environment '$env': torch version = $TORCH_VERSION"
        if [ "$TORCH_VERSION" == "$TARGET_VERSION" ]; then
            echo "  --> Matches the target version!"
        else
            echo "  --> Does not match the target version."
        fi
    else
        echo "Environment '$env': torch is not installed or the environment is invalid."
    fi
    echo "---------------------------------------------"
done

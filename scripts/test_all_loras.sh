#!/bin/bash

# --- 配置 ---
# LoRA 模型所在的目录
# LORA_DIR="datasets/SensorKit/clipdata/finetune_output/00_qwen3_clipdata_text_prepare_sparse03_remove"
LORA_DIR="datasets/SensorKit/clipdata/finetune_output/qwen3_clipdata_text_sparse03_remove"

# 用于存储结果的日志文件
LOG_FILE="lora_test_results.log"

# 初始化最佳模型和分数
BEST_MODEL=""
MAX_FILTERED_FRAMES=-1

# 清空旧的日志文件
> "$LOG_FILE"

echo "Starting LoRA model evaluation..." | tee -a "$LOG_FILE"
echo "Models will be tested from: $LORA_DIR" | tee -a "$LOG_FILE"
echo "-------------------------------------------------" | tee -a "$LOG_FILE"

# 查找所有 .pt 文件并开始循环测试
find "$LORA_DIR" -name "*.pt" | sort | while read -r lora_model_path; do
    echo "--- Testing model: $lora_model_path ---"

    # 1. 运行主定位脚本
    
    # 我们将标准输出和标准错误都重定向，以便后续解析


    OUTPUT=$(python3 -m hloc.pipelines.RobotCar.clip_hloc \
    --caption_model qwen-vl \
    --lora_ckpt_path "$lora_model_path" \
    --text_only_loc \
    --num_loc 1 2>&1)   


    # OUTPUT=$(python3 -m hloc.pipelines.RobotCar.realtime_clip_hloc \
    #     --caption_model qwen-vl \
    #     --lora_ckpt_path "$lora_model_path" \
    #     --text_only_loc \
    #     --exclude_cameras cam1 \
    #     --num_loc 1 2>&1)          

    # 2. 从输出中提取过滤后的帧数
    # 使用 grep 找到包含 "Filtered:" 的行，然后用 sed 提取数字
    FILTERED_COUNT=$(echo "$OUTPUT" | grep "Filtered:" | sed -E 's/.*Filtered: ([0-9]+) frames\./\1/')

    # 检查是否成功提取到数字
    if [[ -z "$FILTERED_COUNT" ]]; then
        echo "ERROR: Could not parse filtered frame count for model $lora_model_path." | tee -a "$LOG_FILE"
        echo "Full output was:" | tee -a "$LOG_FILE"
        echo "$OUTPUT" | tee -a "$LOG_FILE"
        echo "-------------------------------------------------" | tee -a "$LOG_FILE"
        continue
    fi

    # 3. 记录结果并更新最佳模型
    RESULT_LINE="Model: $lora_model_path -> Filtered Frames: $FILTERED_COUNT"
    echo "$RESULT_LINE" | tee -a "$LOG_FILE"

    if (( FILTERED_COUNT > MAX_FILTERED_FRAMES )); then
        MAX_FILTERED_FRAMES=$FILTERED_COUNT
        BEST_MODEL=$lora_model_path
    fi

    echo "-------------------------------------------------" | tee -a "$LOG_FILE"
done

# 4. 打印最终总结报告
echo "" | tee -a "$LOG_FILE"
echo "========== EVALUATION COMPLETE ==========" | tee -a "$LOG_FILE"
if [[ -n "$BEST_MODEL" ]]; then
    echo "Best performing model: $BEST_MODEL" | tee -a "$LOG_FILE"
    echo "Max filtered frames: $MAX_FILTERED_FRAMES" | tee -a "$LOG_FILE"
else
    echo "No models were successfully tested." | tee -a "$LOG_FILE"
fi
echo "=======================================" | tee -a "$LOG_FILE"
echo "Full results have been saved to $LOG_FILE"










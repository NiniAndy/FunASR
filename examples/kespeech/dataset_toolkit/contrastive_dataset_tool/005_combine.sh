#!/bin/bash
phase1=/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Tasks/ASContrastive/tools/text_003A
phase2=/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Tasks/ASContrastive/tools/text_004
output=/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Tasks/ASContrastive/tools/text_005

echo "Combining files $phase1 and $phase2 to $output"
# 合并文件
cat "$phase1" "$phase2" > "$output"

# 显示一条消息表示任务已完成
echo "Done"
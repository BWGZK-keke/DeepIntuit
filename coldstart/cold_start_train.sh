#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

bash scripts/policy_finetune.sh \
    --model_name_or_path "/mnt/bn/themis/kezhang/Qwen2.5-VL-7B-Instruct" \
    --dataset_path "/mnt/bn/themis/kezhang/opensource-code/open_dataset/smarthome_coldstart_train.json" \
    --output_dir "/mnt/bn/themis/kezhang/opensource-code/coldstart_model_test/" \
    --video_max_pixels 200704 \
    --max_frames 32 \
    --gradient_accumulation_steps 64

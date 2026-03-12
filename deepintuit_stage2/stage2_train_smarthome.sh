#!/bin/bash

export USE_U13_FILTER=False

JOB_BASE_PATH=/mnt/bn/themis/kezhang/opensource-code/stage2_model/
mkdir -p "$JOB_BASE_PATH"

PRETRAIN_MODEL_NAME=/mnt/bn/themis/kezhang/opensource-code/stage1_model
QUESTION_PATH=/mnt/bn/themis/kezhang/opensource-code/open_dataset/output_COT_train.json
ANSWER_PATH=$JOB_BASE_PATH/Multihateclip_train_4_answers.json
OUTPUT_PATH=$JOB_BASE_PATH/Multihateclip_train_stage2.json

bash scripts/policy_inference_vllm.sh \
    --saved_model "$PRETRAIN_MODEL_NAME" \
    --question_path "$QUESTION_PATH" \
    --answer_path "$ANSWER_PATH" \
    --max_frames 32 \
    --batch_size 4 \
    --gather_freq 20 \
    --max_new_tokens 2048 \
    --video_max_pixels 200704 \
    --num_workers 32 \
    --num_return_sequences 4

bash scripts/stage2_train_gen.sh \
    --question_path "$QUESTION_PATH" \
    --answer_path "$ANSWER_PATH" \
    --output_path "$OUTPUT_PATH" \
    --mode 'policy_token_with_asr'

AA_ANSWER_FILE=$JOB_BASE_PATH/multihateclip_stage1_infer_out_test.json
TRAIN_DATA_PATH=$OUTPUT_PATH
OUTPUT_DIR=$JOB_BASE_PATH/stage2
AA_EVAL_FILE=/mnt/bn/themis/kezhang/opensource-code/open_dataset/output_COT_test.json
AA_COT_ANSWER_FILE=$OUTPUT_DIR/stage2_multihateclip_infer_out_test_64.json

bash scripts/policy_finetune.sh \
    --model_name_or_path "$PRETRAIN_MODEL_NAME" \
    --dataset_path "$TRAIN_DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --video_max_pixels 200704 \
    --max_frames 32 \
    --num_train_epochs 10

bash scripts/two_stage_inference_policy_token.sh \
    --saved_model "$PRETRAIN_MODEL_NAME" \
    --saved_conf_model "$OUTPUT_DIR" \
    --question_path "$AA_EVAL_FILE" \
    --answer_path "$AA_ANSWER_FILE" \
    --conf_answer_path "$AA_COT_ANSWER_FILE" \
    --video_max_pixels 200704 \
    --batch_size 1 \
    --mode 'policy_token_with_asr' \
    --policy_book_version '20251105' \
    --max_frames 32

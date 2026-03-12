#!/bin/bash
<< 'COMMENT'
bash scripts/nova_infer.sh \
  --saved_model /mnt/bn/themis/gy/sft/hrdd/indicator_wise/speeding/train_pra_pos_2000_neg_2000_model/ \
  --question_path /mnt/bn/themis/gy/sft/hrdd/indicator_wise/speeding/eval_par_pos_400_neg_400.json \
  --answer_path /mnt/bn/themis/gy/sft/hrdd/indicator_wise/speeding/train_pra_pos_2000_neg_2000_model/infer_result.json

COMMENT

# default values
MAX_FRAMES=32
BATCH_SIZE=4
VIDEO_MAX_PIXELS=200704
model_version="qwen"
GATHER_FREQ=100
MAX_NEW_TOKENS=2048
NUM_WORKERS=16
GENERATION_MODE=False
TEMPERATURE="1.0"
DO_SAMPLE=True
NUM_RETURN_SEQUENCES=1
POLICY_TOKEN_VERSION=""

# 参数解析
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --saved_model)
            SAVED_MODEL_PATH="$2"
            shift 2
            ;;
        --saved_conf_model)
            SAVED_CONF_MODEL_PATH="$2"
            shift 2
            ;;
        --model_version)
            model_version="$2"
            shift 2
            ;;
        --question_path)
            QUESTION_PATH="$2"
            shift 2
            ;;
        --answer_path)
            ANSWER_PATH="$2"
            shift 2
            ;;
        --conf_answer_path)
            CONF_ANSWER_PATH="$2"
            shift 2
            ;;
        --video_max_pixels)
            VIDEO_MAX_PIXELS="$2"
            shift 2
            ;;
        --max_frames)
            MAX_FRAMES="$2"
            shift 2
            ;;
        --gather_freq)
            GATHER_FREQ="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 --saved_model <path_to_model> --result_filename <filename> --dataset_path <path> [--selected_indicators <indicators>]"
            exit 1
            ;;
    esac
done


echo "Inferring with accelerate with the following parameters:"
echo "SAVED_MODEL_PATH: $SAVED_MODEL_PATH"
echo "SAVED_CONF_MODEL_PATH: $SAVED_CONF_MODEL_PATH"
echo "QUESTION_PATH: $QUESTION_PATH"
echo "ANSWER_PATH: $ANSWER_PATH"
echo "MAX_FRAMES: $MAX_FRAMES"
echo "VIDEO_MAX_PIXELS: $VIDEO_MAX_PIXELS"

# 1. inference PRA results
bash scripts/policy_inference.sh\
    --saved_model "$SAVED_MODEL_PATH" \
    --question_path "$QUESTION_PATH" \
    --answer_path "$ANSWER_PATH" \
    --max_frames "$MAX_FRAMES" \
    --batch_size "$BATCH_SIZE" \
    --gather_freq "$GATHER_FREQ" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --num_return_sequences 1 \
    --generation_mode "True"

# 2. generate stage2 prompt
COT_PATH="${ANSWER_PATH%.json}_stage2_input.json"
REFORMT_CONF_ANSWER_PATH="${CONF_ANSWER_PATH%.json}_reformat.json"
python3 pipeline/gen_stage2_data.py\
    --answer_file "$ANSWER_PATH" \
    --eval_file "$QUESTION_PATH" \
    --mode "standard" \
    --output_file "$COT_PATH"

# 3. inference confidence model
bash scripts/policy_inference.sh\
    --saved_model "$SAVED_CONF_MODEL_PATH" \
    --question_path "$COT_PATH" \
    --answer_path "$CONF_ANSWER_PATH" \
    --max_frames "$MAX_FRAMES" \
    --batch_size "$BATCH_SIZE" \
    --gather_freq "$GATHER_FREQ" \
    --max_new_tokens 1\
    --num_return_sequences 1 \
    --do_sample "False" \
    --policy_token_version "v1" \
    --generation_mode "True"

# 4. reformat answer
python3 pipeline/reformat_conf_output.py \
    --input_file "$CONF_ANSWER_PATH" \
    --eval_file "$QUESTION_PATH" \
    --output_file "$REFORMT_CONF_ANSWER_PATH" \
    --question_file "$ANSWER_PATH"




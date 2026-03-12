#!/bin/bash
# default values
QUESTION_PATH=''
ANSWER_PATH=''
OUTPUT_PATH=''
MODE='policy_token_asr_with_level'
POLICY_BOOK_VERSION=""

# 参数解析
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --question_path)
            QUESTION_PATH="$2"
            shift 2
            ;;
        --answer_path)
            ANSWER_PATH="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --policy_book_version)
            POLICY_BOOK_VERSION="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 --question_path <training_cot.json> --answer_path <training_infer_result.json>"
            exit 1
            ;;
    esac
done


echo "Generating stage2 training data ..."
echo "Raw Data: $QUESTION_PATH"
echo "After stage 1 inference: $ANSWER_PATH"
echo "Generated stage2 training data path: $OUTPUT_PATH"
echo "mode: $MODE"

# 2. generate stage2 prompt
python3 pipeline/gen_stage2_data.py\
    --answer_file "$ANSWER_PATH" \
    --eval_file "$QUESTION_PATH" \
    --mode "$MODE" \
    --policy_book_version "$POLICY_BOOK_VERSION" \
    --output_file "$OUTPUT_PATH"


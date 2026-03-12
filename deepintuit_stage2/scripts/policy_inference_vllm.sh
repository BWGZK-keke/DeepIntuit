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
GATHER_FREQ=10
MAX_NEW_TOKENS=512
NUM_WORKERS=32
GENERATION_MODE=False
TEMPERATURE="1"
DO_SAMPLE=True
NUM_RETURN_SEQUENCES=2

# 参数解析
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --saved_model)
            SAVED_MODEL_PATH="$2"
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
        --num_return_sequences)
            NUM_RETURN_SEQUENCES="$2"
            shift 2
            ;;
        --do_sample)
            DO_SAMPLE="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --generation_mode)
            GENERATION_MODE="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 --saved_model <path_to_model> --result_filename <filename> --dataset_path <path> [--selected_indicators <indicators>]"
            exit 1
            ;;
    esac
done

# Check required parameters
if [[ -z "$SAVED_MODEL_PATH" || -z "$QUESTION_PATH" || -z "$ANSWER_PATH" ]]; then
    echo "Error: --saved_model, --dataset_path and --result_filename arguments are required."
    echo "Usage: $0 --saved_model <path_to_model> --result_filename <filename> --dataset_path <path> [--selected_indicators <indicators>]"
    exit 1
fi

# generate the up-to-date accelerate config
# num_processes = num_gpus_per_worker * num_workers
NUM_PROCESSES=$(expr $ARNOLD_WORKER_GPU \* $ARNOLD_WORKER_NUM)
MAIN_PROCESS_PORT=${ARNOLD_WORKER_0_PORT:-12345}

# 模型推理（仅使用文件名）
INFER_COMMAND=(
    accelerate launch \
        --num_processes "$NUM_PROCESSES" \
        --main_process_port "$MAIN_PROCESS_PORT" \
        --num_machines "$ARNOLD_WORKER_NUM" \
        --machine_rank "$ARNOLD_ID" \
        --main_process_ip "$ARNOLD_WORKER_0_HOST" \
      nova/runner/nova_vllm_inference.py \
        --saved_model "$SAVED_MODEL_PATH" \
        --model_version "$model_version" \
        --question_path "$QUESTION_PATH" \
        --answer_path "$ANSWER_PATH" \
        --video_max_pixels "$VIDEO_MAX_PIXELS" \
        --max_frames "$MAX_FRAMES" \
        --gather_freq "$GATHER_FREQ" \
        --num_workers "$NUM_WORKERS" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --num_return_sequences "$NUM_RETURN_SEQUENCES" \
        --temperature "$TEMPERATURE"
        --batch_size "$BATCH_SIZE"
)


echo "Inferring with accelerate with the following parameters:"
echo "SAVED_MODEL_PATH: $SAVED_MODEL_PATH"
echo "QUESTION_PATH: $QUESTION_PATH"
echo "ANSWER_PATH: $ANSWER_PATH"
echo "MAX_FRAMES: $MAX_FRAMES"
echo "VIDEO_MAX_PIXELS: $VIDEO_MAX_PIXELS"


if [ -f "${SAVED_MODEL_PATH}/training_args.bin" ]
    then
        # 模型已经训练完了
        echo "###### The model has finished training.#######"
        # 全部训练结束后，清理一下 output_dir 内的 checkpoint
           command_to_run="rm -rf '${SAVED_MODEL_PATH}'/checkpoint-*/"
           echo "About to run: $command_to_run"
           eval "$command_to_run"
    else
        # 模型还没有训练完成，不能推理
        echo "##### The model training has not finished.#####"
        exit 1
fi


if [ -f "$ANSWER_PATH" ]
    then
        # 已经推理完了
        echo "###### The model has finished inferring. #######"
    else
        # Run inference
        "${INFER_COMMAND[@]}"
    sleep 180
fi
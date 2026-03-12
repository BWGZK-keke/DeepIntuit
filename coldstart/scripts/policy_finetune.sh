#!/bin/bash

ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`) # 获取端口号
port=${ports[0]}

# assign default values to command line arguments
max_frames=32
video_max_pixels=200704
freeze_visual_module=True
deepspeed_stage=zero3_offload
num_train_epochs=5
per_device_train_batch_size=1
gradient_accumulation_steps=$((2048 / ${ARNOLD_APPLIED_GPU_NUM}))
model_version="qwen"
save_strategy="steps"
save_steps=100
save_total_limit=2
max_steps=-1
learning_rate=2e-5

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name_or_path)
            model_name_or_path="$2"
            shift 2
            ;;
        --max_steps)
            max_steps="$2"
            shift 2
            ;;
        --save_strategy)
            save_strategy="$2"
            shift 2
            ;;
        --save_steps)
            save_steps="$2"
            shift 2
            ;;
        --save_total_limit)
            save_total_limit="$2"
            shift 2
            ;;
        --dataset_path)
            dataset_path="$2"
            shift 2
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        --max_frames)
            max_frames="$2"
            shift 2
            ;;
        --video_max_pixels)
            video_max_pixels="$2"
            shift 2
            ;;
        --train_config)
            train_config="$2"
            shift 2
            ;;
        --model_version)
            model_version="$2"
            shift 2
            ;;
        --freeze_visual_module)
            freeze_visual_module="$2"
            shift 2
            ;;
        --deepspeed_stage)
            deepspeed_stage="$2"
            shift 2
            ;;
        --num_train_epochs)
            num_train_epochs="$2"
            shift 2
            ;;
        --per_device_train_batch_size)
            per_device_train_batch_size="$2"
            shift 2
            ;;
        --gradient_accumulation_steps)
            gradient_accumulation_steps="$2"
            shift 2
            ;;
        --learning_rate)
            learning_rate="$2"
            shift 2
            ;;
        *)
            echo "Unrecognized argument: $1"
            shift
            ;;
    esac
done

# 检查必需的参数是否已提供
if [[ -z "$model_name_or_path" || -z "$dataset_path" || -z "$output_dir" ]]; then
    echo "Error: Missing required arguments."
    echo "Usage: $0 --model_name_or_path <path_to_model> --dataset_path <path_to_dataset> --output_dir <output_dir>"
    exit 1
fi

DISTRIBUTED_ARGS="
    --nproc_per_node $ARNOLD_WORKER_GPU \
    --nnodes $ARNOLD_WORKER_NUM \
    --node_rank $ARNOLD_ID \
    --master_addr $ARNOLD_WORKER_0_HOST \
    --master_port $port
"

# 执行脚本
echo "Running torchrun with the following parameters:"
echo "model_name_or_path: $model_name_or_path"
echo "model_version: $model_version"
echo "dataset_path: $dataset_path"
echo "output_dir: $output_dir"
echo "max_frames: $max_frames"
echo "video_max_pixels: $video_max_pixels"
echo "num_train_epochs: $num_train_epochs"
echo "freeze_visual_module: $freeze_visual_module"
echo "max_steps: $max_steps"
echo "save_strategy: $save_strategy"
echo "save_steps: $save_steps"
echo "save_total_limit: $save_total_limit"
echo "gradient_accumulation_steps: $gradient_accumulation_steps"
echo "learning_rate: $learning_rate"

if [ -f "$output_dir/training_args.bin" ]
  then
      # 模型已经训练完了
      echo "###### The model has finished training. Skip training #######"
  else
      # 开始训练
      torchrun $DISTRIBUTED_ARGS nova/runner/nova_finetune.py \
          --model_name_or_path "$model_name_or_path" \
          --model_version "$model_version" \
          --num_train_epochs $num_train_epochs \
          --max_steps $max_steps \
          --per_device_train_batch_size $per_device_train_batch_size \
          --per_device_eval_batch_size 1 \
          --gradient_accumulation_steps $gradient_accumulation_steps \
          --eval_strategy "no" \
          --learning_rate $learning_rate \
          --logging_steps 1 \
          --warmup_ratio 0.06 \
          --deepspeed "nova/configs/deepspeed_configs/$deepspeed_stage.json" \
          --fp16 False \
          --bf16 True \
          --save_strategy "$save_strategy" \
          --save_steps $save_steps \
          --save_total_limit $save_total_limit \
          --remove_unused_columns False \
          --dataset_path "$dataset_path" \
          --output_dir "$output_dir" \
          --max_frames "$max_frames" \
          --video_max_pixels "$video_max_pixels" \
          --freeze_visual_module "$freeze_visual_module" \
          --gradient_checkpointing True \
          --report_to none \
          --dataloader_num_workers 32 \
          --dataloader_prefetch_factor 3 \
          --dataloader_pin_memory True

          # $( [ "$freeze_visual_module" == "true" ] && echo "--freeze_visual_module" )
      sleep 180
fi

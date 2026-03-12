#!/bin/bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 -m recipe.nova.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/bn/themis/kezhang/opensource-code/open_dataset/stage1_train.json \
    data.val_files=/mnt/bn/themis/kezhang/opensource-code/open_dataset/stage1_train.json \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=26000 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.dataloader_num_workers=32 \
    data.truncation='error' \
    data.video_key=videos \
    actor_rollout_ref.model.path=/mnt/bn/themis/kezhang/opensource-code/coldstart_model \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.tis_imp_ratio_cap=2.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_model_len=128000 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=60000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    +actor_rollout_ref.rollout.stop_token_ids='[2, 13]' \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.video=1 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_lsj' \
    trainer.experiment_name='qwen_stage1' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.log_val_generations=-1 \
    trainer.val_before_train=False \
    trainer.default_local_dir="/mnt/bn/themis/kezhang/opensource-code/stage1_grpo_smarthome" \
    trainer.validation_data_dir="/mnt/bn/themis/kezhang/opensource-code/stage1_grpo_smarthome/val" \
    trainer.total_epochs=20 \
    custom_reward_function.name='violation_bullying_compute_score'

# you need to check if the search directory is correct. (line 61 in stage1_opensource/recipe/nova/merge_model.sh)
python3 recipe/nova/merge_model.py stage1_grpo_smarthome /mnt/bn/themis/kezhang/opensource-code

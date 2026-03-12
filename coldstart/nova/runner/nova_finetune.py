import os
import json
import pathlib
import re
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
from transformers import Trainer, TrainingArguments, HfArgumentParser

PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/mnt/bn/themis/data/LLM/vegeta_v1b_siglip_mistral_stage2_0923")
    model_version: str = field(default="")
    freeze_visual_module: Optional[bool] = field(default=False)

@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    dataset_path: str = field(default="")
    max_frames: int = field(default=32)


@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: Optional[str] = field(default="")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False

    fp16: Optional[bool] = field(default=False)
    bf16: Optional[bool] = field(default=False)

    seed = 42

    is_inter_eval: Optional[bool] = field(default=False)
    eval_step: int = 500
    video_max_pixels: int = 200704

def main(training_args, model_args, data_args):
    print("model_version: ", model_args.model_version)

    from nova.model.nova_qwen.modeling_nova import NovaForConditionalGeneration
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from nova.model.nova_qwen.configuration_nova import NovaConfig
    from nova.model.nova_qwen.processing_nova import NovaProcessor
    from nova.data_utils.nova_qwen_dataset import NovaDataset, NovaCollator
    
    if os.path.basename(model_args.model_name_or_path) == "config.json":
    # Case 1: The path is directly to a config file — load config only, no weights
        config_dir = os.path.dirname(model_args.model_name_or_path)
        config = NovaConfig.from_pretrained(config_dir)
        config._attn_implementation = "flash_attention_2"
        model = NovaForConditionalGeneration(
            config=config
        ).to(dtype=torch.bfloat16)
        processor = NovaProcessor.from_pretrained(config_dir)
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        processor = NovaProcessor.from_pretrained(model_args.model_name_or_path)
    processor.image_processor.video_max_pixels = training_args.video_max_pixels

    # add multi-label support if policy_info.json is detected in the same directory where dataset resides
    policy_info_local_path = os.path.join(os.path.dirname(data_args.dataset_path), "policy_info.json")
    if os.path.exists(policy_info_local_path):
        print(f"Path info file **exist**: {policy_info_local_path}")

        policy_tokens = ['<NO_POLICY>']
        policy_info = json.load(open(policy_info_local_path))

        # 强制有序
        policy_info_sorted = sorted(list(policy_info.items()))
        for line in policy_info_sorted:
            policy_tokens.append(line[1][0]['Policy Token'])

    else:
        print(f"Path info file **missing**: {policy_info_local_path}")
        policy_tokens = ["Approve", "Violation"]

    # add indicator QA token support if ind_token_dict.json is detected in the same directory where dataset resides
    # New version for all policy: add indicator QA token support if ind_token_dict.json is detected in the same directory where dataset resides
    # ind_tokens are already sorted
    combined_tokens = policy_tokens.copy()
    ind_token_local_path = os.path.join(os.path.dirname(data_args.dataset_path), "ind_token_dict.json")
    if os.path.exists(ind_token_local_path):
        print(f"Path indicator token file **exist**: {ind_token_local_path}")
        with open(ind_token_local_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Accept either a plain list [...], or {"additional_special_tokens": [...]}
        if isinstance(raw, list):
            ind_tokens = raw
        elif isinstance(raw, dict) and "additional_special_tokens" in raw and isinstance(raw["additional_special_tokens"], list):
            ind_tokens = raw["additional_special_tokens"]
        else:
            print("Unexpected ind_token_dict.json format; expected a list or {'additional_special_tokens': [...]} . Skipping.")
            ind_tokens = []

        # Keep only non-empty strings
        ind_tokens = [t for t in ind_tokens if isinstance(t, str) and t.strip()]

        # Append to policy_tokens without introducing duplicates, preserving order
        seen = set(combined_tokens)
        for t in ind_tokens:
            if t not in seen:
                combined_tokens.append(t)
                seen.add(t)
    else:
        print("Indicator token file **missing**")

    combined_tokens.extend(['<think>', '</think>','<answer>','</answer>'])

    new_special_tokens = {
        'additional_special_tokens': combined_tokens
    }

    processor.tokenizer.add_special_tokens(new_special_tokens)
    model.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)

    # Get back the token ids
    policy_token_ids = processor.tokenizer.convert_tokens_to_ids(policy_tokens)

    # Freeze visual module parameters if freeze_visual_module is True
    if model_args.freeze_visual_module:
        for param in model.visual.parameters():
            param.requires_grad = False

    dataset = NovaDataset(data_args.dataset_path, max_frames=data_args.max_frames)

    # Train on a subset of the dataset for debugging
    # from torch.utils.data import Subset
    # dataset = Subset(dataset, range(10000))

    data_collator = NovaCollator(data_type='bf16', processor=processor)

    if dist.is_initialized():
        global_rank = dist.get_rank()
    else:
        global_rank = 0

    # Initialize wandb only in the main process
    # if global_rank == 0 and os.environ.get("CLOUDNATIVE_CLUSTER", "") != "cloudnative-syd1a":
    #     wandb.init(project="Nova_SFT", group="DDP")

    ## unweighted
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # Start training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model and processor
    trainer.save_model(training_args.output_dir)
    if global_rank == 0:
        processor.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    parser = HfArgumentParser(
        (TrainingArguments, ModelArguments, DataArguments)
    )

    training_args, model_args, data_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    main(training_args, model_args, data_args)

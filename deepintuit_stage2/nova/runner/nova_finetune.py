import os
import json
import os
import pathlib
import re
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
from transformers import Trainer, TrainingArguments, HfArgumentParser
import warnings
def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    model_version: str = field(default="")
    freeze_visual_module: Optional[bool] = field(default=False)
    model_type: str = field(default="")
    problem_type: str = field(default="multi_label_classification")
    label_type: str = field(default="")

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
    # deepspeed: str = field(default='./configs/zero3.json')

import random
from collections import Counter

IGNORE_TOKEN_ID = -100

def _safe_decode(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False)

def debug_one_batch(model, processor, data_collator, dataset, device=None, n=2, max_chars=800,
                    do_forward=True, forward_fp32=False, low_supervised_ratio_warn=0.001):
    """
    Runs collator on n samples, prints masking stats + decoded supervised text,
    and checks:
      - whether model.forward is using your labels (by comparing loss w/ real labels vs all -100)
      - whether loss/logits are finite
      - supervised token counts in the actual batch
    """
    import random
    import torch
    import torch.nn.functional as F

    IGNORE = IGNORE_TOKEN_ID
    tok = processor.tokenizer

    def _find_last_subsequence(seq, pattern):
        if not pattern or len(pattern) > len(seq):
            return -1
        last = -1
        for i in range(len(seq) - len(pattern) + 1):
            if seq[i:i + len(pattern)] == pattern:
                last = i
        return last

    def _find_first_subsequence_from(seq, pattern, start):
        if not pattern or len(pattern) > len(seq):
            return -1
        for i in range(start, len(seq) - len(pattern) + 1):
            if seq[i:i + len(pattern)] == pattern:
                return i
        return -1

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # sample a few items
    idxs = [random.randrange(len(dataset)) for _ in range(n)]
    features = [dataset[i] for i in idxs]
    batch = data_collator(features)

    print("\n===== DEBUG BATCH KEYS =====")
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"{k:22s} shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
        else:
            print(f"{k:22s} type={type(v)}")

    if "input_ids" not in batch:
        print("No input_ids in batch.")
        return

    input_ids = batch["input_ids"]
    labels = batch.get("labels", None)
    attn = batch.get("attention_mask", None)

    print("\n===== TOKENIZER IDS =====")
    for t in ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
              "<|video_pad|>", "<|image_pad|>", "<|vision_pad|>"]:
        if t in tok.get_vocab():
            print(f"{t:16s} -> {tok.convert_tokens_to_ids(t)}")
    print("pad_token_id:", tok.pad_token_id, " pad_token:", tok.pad_token)
    print("eos_token_id:", tok.eos_token_id, " eos_token:", tok.eos_token)
    print("bos_token_id:", tok.bos_token_id, " bos_token:", tok.bos_token)

    # dtype sanity
    print("\n===== DTYPE SANITY =====")
    print("input_ids dtype:", input_ids.dtype, "(should be torch.long)")
    if labels is not None and torch.is_tensor(labels):
        print("labels dtype:   ", labels.dtype, "(should be torch.long)")
    if attn is not None and torch.is_tensor(attn):
        print("attn dtype:     ", attn.dtype)

    print("\n===== DEBUG MASK STATS =====")
    if labels is None:
        print("No labels in batch -> Trainer won't compute LM loss.")
        return

    if input_ids.shape != labels.shape:
        print("WARNING: input_ids.shape != labels.shape", tuple(input_ids.shape), tuple(labels.shape))

    total = labels.numel()
    ignored = (labels == IGNORE).sum().item()
    ignored_ratio = ignored / total if total else 0.0
    print(f"total label tokens: {total}")
    print(f"ignored (-100):     {ignored} ({ignored_ratio:.6f})")

    keep = (labels != IGNORE)
    sup_counts = keep.sum(dim=1).tolist()
    supervised_ratio = float(keep.float().mean().item())
    print("supervised tokens per sample:", sup_counts[: min(len(sup_counts), 16)])
    print("supervised ratio:", supervised_ratio)
    print("zero-supervised samples:", sum(1 for c in sup_counts if c == 0))

    assistant_prefix_ids = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    assistant_end_ids = tok.encode("<|im_end|>", add_special_tokens=False)
    missing_prefix_count = 0
    missing_end_count = 0
    assistant_spans = []
    for i in range(input_ids.size(0)):
        ids_i = input_ids[i].tolist()
        prefix_start = _find_last_subsequence(ids_i, assistant_prefix_ids)
        if prefix_start == -1:
            missing_prefix_count += 1
            assistant_spans.append((i, -1, -1))
            continue
        content_start = prefix_start + len(assistant_prefix_ids)
        content_end = _find_first_subsequence_from(ids_i, assistant_end_ids, content_start)
        if content_end == -1:
            missing_end_count += 1
        assistant_spans.append((i, content_start, content_end))

    print("assistant span parse: missing_prefix_count =", missing_prefix_count)
    print("assistant span parse: missing_im_end_count =", missing_end_count)
    print("assistant spans sample (batch_idx, start, end):", assistant_spans[: min(len(assistant_spans), 8)])

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank == 0 and supervised_ratio < low_supervised_ratio_warn:
        print(
            f"[WARN] supervised ratio {supervised_ratio:.6f} < {low_supervised_ratio_warn:.6f}; "
            "please verify assistant-span masking."
        )

    non_ign = labels[keep]
    if non_ign.numel() > 0:
        min_id = int(non_ign.min().item())
        max_id = int(non_ign.max().item())
        full_vocab = len(tok)  # IMPORTANT: includes added tokens
        print(f"non-ignored labels: count={non_ign.numel()} min={min_id} max={max_id} vocab_size(len)={full_vocab}")
        if min_id < 0 or max_id >= full_vocab:
            print("WARNING: found non-ignored label id outside tokenizer len(tokenizer)!")

        bad_pad = int((non_ign == tok.pad_token_id).sum().item()) if tok.pad_token_id is not None else 0
        bad_eos = int((non_ign == tok.eos_token_id).sum().item()) if tok.eos_token_id is not None else 0
        print(f"NON-IGNORED pad count: {bad_pad}")
        print(f"NON-IGNORED eos count: {bad_eos}")

        vision_tokens = ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>", "<|vision_pad|>"]
        vision_ids = [tok.convert_tokens_to_ids(t) for t in vision_tokens if t in tok.get_vocab()]
        if vision_ids:
            vid = torch.tensor(vision_ids, device=labels.device)
            non_ign_is_vision = int(torch.isin(non_ign, vid).sum().item())
            print("vision token ids:", vision_ids)
            print(f"NON-IGNORED vision-token count: {non_ign_is_vision}")

        non_ign_unique = torch.unique(non_ign[: min(non_ign.numel(), 20000)])
        print("non-ignored unique labels (sample):", non_ign_unique[:50].tolist())
    else:
        print("No supervised tokens in this batch (all labels are -100).")

    print("\n===== DECODE SAMPLE(S) =====")
    bsz = input_ids.shape[0]
    for i in range(min(bsz, n)):
        ids = input_ids[i]
        lab = labels[i]
        keep_i = (lab != IGNORE)

        full_txt = processor.tokenizer.decode(ids.tolist(), skip_special_tokens=False)
        sup_txt = processor.tokenizer.decode(ids[keep_i].tolist(), skip_special_tokens=False)

        print(f"\n--- sample {i} (dataset idx {idxs[i]}) ---")
        print("input decoded (first chars):")
        print(full_txt[:])
        print("\nsupervised decoded (input_ids where labels!=-100) (first chars):")
        print(sup_txt[:])
        print("supervised token count:", int(keep_i.sum().item()))
        if attn is not None:
            print("attn sum:", int(attn[i].sum().item()))


def main(training_args, model_args, data_args):
    print("model_version: ", model_args.model_version)
    from transformers import Qwen2_5_VLForConditionalGeneration as NovaForConditionalGeneration
    from nova.model.nova_qwen.processing_nova import NovaProcessor as NovaProcessor
    from nova.data_utils.nova_mistral_dataset import NovaDataset, NovaCollator

    model = NovaForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    if model_args.model_type == 'seq_cls':
        model.config.problem_type = model_args.problem_type

    # ---- IMPORTANT: build processor first and use its tokenizer ----
    processor = NovaProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    tokenizer = processor.tokenizer  # single source of truth

    # If you really must replace tokenizer, do it here, but ensure it's consistent:
    # tokenizer = hf_tokenizer(model_args.model_name_or_path, trust_remote_code=True)
    # processor.tokenizer = tokenizer

    processor.image_processor.video_max_pixels = training_args.video_max_pixels

    # ---- load policy tokens ----
    policy_info_local_path = os.path.join(os.path.dirname(data_args.dataset_path), "policy_info.json")
    policy_info = json.load(open(policy_info_local_path))
    policy_info_sorted = sorted(list(policy_info.items()))

    policy_tokens = []
    policy_token_2_policy_idx = {}
    for i, line in enumerate(policy_info_sorted):
        tok_str = line[1][0]['Policy Token']
        policy_tokens.append(tok_str)
        policy_token_2_policy_idx[tok_str] = i

    # ---- add special tokens to tokenizer and resize model ----
    # tokenizer.add_special_tokens({"additional_special_tokens": policy_tokens})

    # # Resize BOTH input embeddings and lm_head to match tokenizer length
    # new_n = len(tokenizer)
    # model.resize_token_embeddings(new_n)

    # model.config.vocab_size = new_n
    # if hasattr(model, "vocab_size"):
    #     model.vocab_size = new_n

    model.config.use_cache = False

    # Sanity prints (do once)
    print("len(tokenizer):", len(tokenizer))
    print("tokenizer.vocab_size (base):", tokenizer.vocab_size)
    print("model input embed:", model.get_input_embeddings().weight.shape[0])
    if model.get_output_embeddings() is not None:
        print("model output embed:", model.get_output_embeddings().weight.shape[0])

    # ---- Freeze visual module parameters if needed ----
    if model_args.freeze_visual_module:
        for param in model.visual.parameters():
            param.requires_grad = False

    dataset = NovaDataset(data_args.dataset_path, max_frames=data_args.max_frames,train_mode=True)

    if model_args.model_type == 'seq_cls':
        data_collator = NovaSeqClsCollator(
            data_type='bf16',
            processor=processor,
            policy_token_2_policy_idx=policy_token_2_policy_idx,
            label_type=model_args.label_type
        )
    else:
        data_collator = NovaCollator(data_type='bf16', processor=processor,train_mode=True)

    if dist.is_available() and dist.is_initialized():
        global_rank = dist.get_rank()
    else:
        global_rank = 0

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    #debug_one_batch(model, processor, data_collator, dataset, device=model.device, n=2)
    # If you want to stop after debugging:
    # raise SystemExit("Stopping after debug.")
    # Start training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # # # # #Save model and processor
    trainer.save_model(training_args.output_dir)
    if global_rank == 0:
        processor.save_pretrained(training_args.output_dir)

    if global_rank == 0:
        print("Finished!")
        # uploading to tos sometimes fails without an obvious reason
        # try:
        #     # upload_saved_model_to_tos(training_args.output_dir)
        # except Exception as e:
        #     print(f"Failed to upload model to tos. Error: {e}")
if __name__ == "__main__":
    parser = HfArgumentParser(
        (TrainingArguments, ModelArguments, DataArguments)
    )

    training_args, model_args, data_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    main(training_args, model_args, data_args)

from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from recipe.nova.rl_dataset import RLHFDataset as NovaDataset
import torch
from torch.utils.data import RandomSampler, DataLoader
from recipe.nova.model.processing_nova import NovaProcessor
from verl.protocol import DataProto
from omegaconf import OmegaConf, DictConfig
import numpy as np
from vllm import ModelRegistry, LLM, SamplingParams
from recipe.nova.model.modeling_nova_vllm import NovaForConditionalGeneration 
from recipe.nova.model.configuration_nova import NovaConfig
from recipe.nova.model.image_processing_nova import NovaImageProcessor
from recipe.nova.model.processing_nova import NovaProcessor
from transformers import CONFIG_MAPPING, IMAGE_PROCESSOR_MAPPING, PROCESSOR_MAPPING
from transformers import Qwen2_5_VLForConditionalGeneration

CONFIG_MAPPING.register('nova', NovaConfig)
IMAGE_PROCESSOR_MAPPING.register('NovaImageProcessor', (NovaImageProcessor, NovaImageProcessor))
PROCESSOR_MAPPING.register('NovaProcessor', (NovaProcessor, NovaProcessor))
# ModelRegistry.register_model("NovaForConditionalGeneration", NovaForConditionalGeneration)
ModelRegistry.register_model("NovaForConditionalGeneration", Qwen2_5_VLForConditionalGeneration)


MIN_PIXELS=262144
MAX_PIXELS=4194304

if __name__ == '__main__':
    model_path = '/mnt/bn/themis/kezhang/opensource-code/coldstart_model/'
    dataset_path = '/mnt/bn/themis/kezhang/opensource-code/open_dataset/stage1_train.json'
    config =  OmegaConf.create({})

    from verl.utils import hf_processor, hf_tokenizer
    trust_remote_code = True
    #tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
    # tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
    tokenizer = hf_tokenizer("path to Qwen2.5-VL-7B-Instruct", trust_remote_code=trust_remote_code)
    # Used for multimodal LLM, could be None
    # processor = hf_processor(model_path, trust_remote_code=trust_remote_code, use_fast=True)
    processor = NovaProcessor.from_pretrained(model_path)
    # processor.image_processor.max_pixels = 254016
    # processor.image_processor.video_max_pixels = 254016
    # tokenizer = processor.tokenizer

    valid_dataset = NovaDataset(dataset_path, tokenizer=tokenizer, processor=processor, max_frames=32, config=config, train_mode=False)
    data_loader = DataLoader(valid_dataset, batch_size=1, collate_fn=collate_fn, num_workers=0, shuffle=False)

    sampling_params = SamplingParams(
        temperature=0.01,
        max_tokens=1024,
        stop_token_ids=[],
        logprobs=20,
        n=1,
        skip_special_tokens=False,
    )
    mm_kwargs = {
        'video_max_pixels': 200072,
    }

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,  # 你想用 Accelerate 做数据并行，所以 vLLM 内部的张量并行设为 1
        # 其他参数
        gpu_memory_utilization=0.7,
        max_num_seqs=2,
        max_num_batched_tokens=40000, 
        enforce_eager=True 
    )

    for item2 in data_loader:
        prompts = DataProto.from_single_dict(item2)
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        # position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        # eos_token_id = prompts.meta_info["eos_token_id"]
        eos_token_id = tokenizer.eos_token_id
        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")
        # breakpoint()
        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data, "mm_processor_kwargs": mm_kwargs})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        for input_data in vllm_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])
        print(input_data["multi_modal_data"]["video"][0].shape)
        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
        print(outputs[0].outputs[0].text)
        # # print(outputs[1].outputs[0].text)
        # print('---------')
    






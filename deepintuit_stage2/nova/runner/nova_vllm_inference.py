import argparse
import json
import os
import torch
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import StoppingCriteria
from vllm import LLM, SamplingParams
from vllm import ModelRegistry
from vllm.model_executor.models.llama import LlamaModel
from nova.model.nova_qwen.configuration_nova import NovaConfig
from nova.model.nova_qwen.image_processing_nova import NovaImageProcessor
from nova.model.nova_qwen.processing_nova import NovaProcessor
from nova.model.nova_qwen.modeling_nova import NovaForConditionalGeneration as NovaForConditionalGeneration
from transformers import CONFIG_MAPPING, IMAGE_PROCESSOR_MAPPING, PROCESSOR_MAPPING
import torch.profiler
import time
import warnings
CONFIG_MAPPING.register('nova', NovaConfig)
IMAGE_PROCESSOR_MAPPING.register('NovaImageProcessor', (NovaImageProcessor, NovaImageProcessor))
PROCESSOR_MAPPING.register('NovaProcessor', (NovaProcessor, NovaProcessor))
from transformers import Qwen2_5_VLForConditionalGeneration
ModelRegistry.register_model("NovaForConditionalGeneration", Qwen2_5_VLForConditionalGeneration)

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



class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence=[2]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids


def encode_strings(strings):
    """将字符串列表编码为字节数组列表"""
    return [list(s.encode('utf-8')) for s in strings]


def decode_string(byte_array):
    """将字节数组列表解码为字符串列表"""
    return ''.join([chr(b) for b in byte_array])

def main(args):
    tokenizer = hf_tokenizer("/mnt/bn/themis/kezhang/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)


    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        skip_special_tokens=False,
        logprobs=20,
        n=args.num_return_sequences
    )

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"Running data-parallel inference on {accelerator.num_processes} GPUs using Accelerate.")
        print("Results will be synchronized in memory.")
    # 3. 初始化 vLLM，并传入 
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        torch.cuda.set_device(accelerator.device)
    os.environ["RANK"] = str(accelerator.process_index)
    os.environ["LOCAL_RANK"] = str(accelerator.local_process_index)
    os.environ["WORLD_SIZE"] = str(accelerator.num_processes)
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12345")

    llm = LLM(
        model=args.saved_model,
        tensor_parallel_size=1,  # 你想用 Accelerate 做数据并行，所以 vLLM 内部的张量并行设为 1
        # 其他参数
        mm_processor_kwargs={
            #   "video_max_pixels": args.video_max_pixels,
              "max_pixels": args.video_max_pixels
        },
        gpu_memory_utilization=0.7,
        max_num_seqs=args.batch_size * args.num_return_sequences,
        max_num_batched_tokens=50000, 
        enforce_eager=False, 
        distributed_executor_backend="external_launcher",
        seed=accelerator.process_index,
        trust_remote_code=True,
    )

    if args.model_version == "mistral":
        from nova.model.nova_mistral.modeling_nova import NovaForConditionalGeneration as NovaForConditionalGeneration
        from nova.model.nova_mistral.processing_nova import NovaProcessor as NovaProcessor
        from nova.data_utils.nova_mistral_dataset import NovaDataset, NovaCollator, NovaVllmCollator
    elif args.model_version == "qwen":
        from nova.model.nova_qwen.modeling_nova import NovaForConditionalGeneration as NovaForConditionalGeneration
        from nova.model.nova_qwen.processing_nova import NovaProcessor as NovaProcessor
        from nova.data_utils.nova_mistral_dataset import NovaDataset, NovaCollator, NovaVllmCollator
    # processor = AutoImageProcessor.from_pretrained(args.saved_model, trust_remote_code=True)
    if dist.is_initialized():
        global_rank = dist.get_rank()
    else:
        global_rank = 0

    # add multi-label support
    policy_info_local_path = os.path.join(os.path.dirname(args.question_path), "policy_info.json")
    # if os.path.exists(policy_info_local_path):
    print(f"Path info file **exist**: {policy_info_local_path}")

    # policy_tokens = ['<NO_POLICY>']
    policy_tokens = []
    with open(policy_info_local_path, 'r', encoding='utf-8') as f:
        policy_info = json.load(f)

    # 强制有序
    policy_info_sorted = sorted(list(policy_info.items()))
    for line in policy_info_sorted:
        policy_tokens.append(line[1][0]['Policy Token'])

    processor = NovaProcessor.from_pretrained(args.saved_model)
    # processor.image_processor.max_pixels = args.video_max_pixels
    # processor.tokenizer = tokenizer

    # 数据部分
    dataset = NovaDataset(args.question_path, max_frames=args.max_frames, train_mode=False)
    data_collator = NovaVllmCollator(processor=processor, tokenizer=tokenizer, train_mode=False)
    #data_collator = NovaVllmCollator(data_type='bf16', processor=processor, train_mode=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator, num_workers=args.num_workers, prefetch_factor=3)

    indices = [processor.tokenizer.encode(t)[1] if processor.tokenizer.encode(t)[0] == processor.tokenizer.bos_token_id
            else processor.tokenizer.encode(t)[0] for t in policy_tokens]
    dataloader = accelerator.prepare(dataloader)

    res_local = []
    pbar = tqdm(dataloader, disable=not accelerator.is_main_process, desc="Processing batches")
    batch_count = 0
    for data in pbar:
        vllm_inputs = []
        for raw_prompt_ids, multi_modal_data in zip(
            data["raw_prompt_ids"], data["multi_modal_data"]):
            vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
            # vllm_inputs.append({"prompt": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        # with torch.profiler.profile(
        #     activities=[
        #         torch.profiler.ProfilerActivity.CPU,
        #         torch.profiler.ProfilerActivity.CUDA,
        #     ],
        #     record_shapes=True,
        #     with_stack=True
        # ) as prof:
        # print("Video frames length:", len(vllm_inputs[0]["multi_modal_data"]["video"][0]))
        # print(vllm_inputs[0])
        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
        # print("############################")
        # print(outputs[0].outputs[0].text)
        for i in range(args.batch_size):
            for j in range(args.num_return_sequences):
                token_log_probs = outputs[i].outputs[j].logprobs[0]
                score = []
                for indice in indices:
                    if indice in token_log_probs:
                        score.append(token_log_probs[indice].logprob)
                    else:
                        score.append(float('-inf'))
                score = torch.tensor(score)
                # accelerator.print(outputs[i].outputs[j].text)
                # accelerator.print(data['item_ids'][i])
                # accelerator.print('---------------------')
                line = {
                    'item_id': data['item_ids'][i],
                    'response': outputs[i].outputs[j].text, 
                    'score': score.tolist()
                }
                # print(line['item_id'])
                # print(line['response'])
                res_local.append(line)

        # vllm_end_time = time.time()
        # print(f"Data loading time: {data_loading_end_time - iter_start_time}")
        # print(f"vLLM inference time: {vllm_end_time - data_loading_end_time}")
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        # iter_start_time = time.time() # 在循环末尾更新
        batch_count += 1
        if batch_count % args.gather_freq == 0:
            # accelerator.print(res_local[-4:])
            accelerator.wait_for_everyone()

    temp_output_path = f"{args.answer_path}.part_{accelerator.process_index}.json"
    with open(temp_output_path, 'w', encoding='utf-8') as f:
        json.dump(res_local, f, ensure_ascii=False, indent=4)     

    accelerator.wait_for_everyone()
    # 主进程 (is_main_process) 负责合并所有临时文件
    if accelerator.is_main_process:
        print("All processes finished. Merging results...")
        final_results = []
        for r in range(accelerator.num_processes):
            part_path = f"{args.answer_path}.part_{r}.json"
            try:
                with open(part_path, 'r', encoding='utf-8') as f:
                    part_data = json.load(f)
                    final_results.extend(part_data)
                os.remove(part_path) # 合并后删除临时文件
            except FileNotFoundError:
                print(f"Warning: Temporary file {part_path} not found. Skipping.")
        # 将最终结果写入指定的输出文件
        with open(args.answer_path, 'w', encoding='utf-8') as f1:
            for line in final_results:
                f1.write(json.dumps(line, ensure_ascii=False) + "\n")
                f1.flush()

        print(f"Inference complete. Results saved to {args.answer_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run VLM model inference.")
    parser.add_argument("--saved_model", type=str,
                        default='',
                        help="Path to the saved model directory.")

    parser.add_argument("--model_version", type=str, default="mistral", choices=["mistral", "qwen", "falcon"],
                        help="the version of Vegeta Models to use. Select from ['mistral', 'qwen']")

    parser.add_argument("--question_path", type=str,
                        default="",
                        help="Path to the dataset file.")

    parser.add_argument("--answer_path", type=str,
                        default='debug/20250611_infer_vllm_debug.json',
                        help="Filename of to model result file.")

    parser.add_argument("--max_frames", type=int, default=64,
                        help="Maximum number of sampled frames for each video.")

    parser.add_argument("--video_max_pixels", type=int, default=254016,
                        help="Maximum number of pixels for each frame in the video.")

    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference.")

    parser.add_argument("--gather_freq", type=int, default=10,
                        help="sync frequency between different GPUs.")

    parser.add_argument("--num_workers", type=int, default=16,
                        help="loader num_workers.")

    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="max new tokens for LLM output.")

    parser.add_argument("--num_return_sequences", type=int, default=2,
                        help="num_return_sequence")
    
    parser.add_argument("--temperature", type=float, default=0.01,
                        help="temperature.")
                        
    args = parser.parse_args()

    main(args)
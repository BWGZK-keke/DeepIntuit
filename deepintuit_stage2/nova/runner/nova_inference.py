import argparse
import json
import os

import torch
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import StoppingCriteria


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        # 这会向用户显示一个清晰的错误信息
        raise argparse.ArgumentTypeError('Boolean value expected.')

class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
        return all(seq == self.eos_sequence for seq in last_ids)


def encode_strings(strings):
    """将字符串列表编码为字节数组列表"""
    return [list(s.encode('utf-8')) for s in strings]


def decode_string(byte_array):
    """将字节数组列表解码为字符串列表"""
    return ''.join([chr(b) for b in byte_array])

def main(args):
    if args.model_version == "mistral":
        from nova.model.nova_mistral.modeling_nova import NovaForConditionalGeneration as NovaForConditionalGeneration
        from nova.model.nova_mistral.processing_nova import NovaProcessor as NovaProcessor
        from nova.data_utils.nova_mistral_dataset import NovaDataset, NovaCollator
    elif args.model_version == "qwen":
        from nova.model.nova_qwen.modeling_nova import NovaForConditionalGeneration as NovaForConditionalGeneration
        from nova.model.nova_qwen.processing_nova import NovaProcessor as NovaProcessor
        from nova.data_utils.nova_mistral_dataset import NovaDataset, NovaCollator, NovaVllmCollator
    from transformers import Qwen2_5_VLForConditionalGeneration
    print(args)
    accelerator = Accelerator()

    if dist.is_initialized():
        global_rank = dist.get_rank()
    else:
        global_rank = 0

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.saved_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda"
    )
    model.eval()

    # add multi-label support
    # if args.policy_token_version == '':
    policy_info_local_path = os.path.join(os.path.dirname(args.question_path), "policy_info.json")
    if not os.path.exists(policy_info_local_path):
        policy_info_local_path = os.path.join(os.path.dirname(os.path.dirname(args.question_path)), "policy_info.json")
    
    # print(policy_info_local_path)
    # print("#####################")
    # if os.path.exists(policy_info_local_path):
    print(f"Path info file **exist**: {policy_info_local_path}")

    # policy_tokens = ['<NO_POLICY>']
    policy_tokens = []
    policy_info = json.load(open(policy_info_local_path))

    # 强制有序
    policy_info_sorted = sorted(list(policy_info.items()))
    for line in policy_info_sorted:
        policy_tokens.append(line[1][0]['Policy Token'])

    processor = NovaProcessor.from_pretrained(args.saved_model)
    tokenizer = processor.tokenizer
    # tokenizer.add_special_tokens({"additional_special_tokens": policy_tokens})
    processor.image_processor.video_max_pixels = args.video_max_pixels
    # 数据部分
    dataset = NovaDataset(args.question_path, max_frames=args.max_frames, train_mode=False)
    data_collator = NovaCollator(data_type='bf16', processor=processor, train_mode=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator, num_workers=args.num_workers, prefetch_factor=2)

    # 开始推理
    model, dataloader = accelerator.prepare(model, dataloader)

    def gather_move_clear(gpu_lists, cpu_lists):
        for gpu_list, cpu_list in zip(gpu_lists, cpu_lists):
            gathered = accelerator.gather(torch.cat(gpu_list, dim=0)).cpu()
            cpu_list.append(gathered)
            gpu_list.clear()
        torch.cuda.empty_cache()

    all_generated_gpu, all_item_ids_gpu, all_scores_gpu = [], [], []
    all_generated_cpu, all_item_ids_cpu, all_scores_cpu = [], [], []
    batch_count = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            # try:
            input_data = {k: v for k, v in data.items() if k not in ('item_ids', 'labels')}
            # print(input_data["input_ids"])
            # for seq in input_data["input_ids"]:
            #     print(tokenizer.decode(seq, skip_special_tokens=False))
            if hasattr(model, 'module'):
                model = model.module
            if args.generation_mode:
                outputs = model.generate(
                    **input_data,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    num_return_sequences=args.num_return_sequences,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    
                )
            else:
                eos_id = processor.tokenizer.eos_token_id
                eos_sequence = [eos_id] if eos_id is not None else []
                outputs = model.generate(
                    **input_data,
                    max_new_tokens=args.max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                    stopping_criteria=[EosListStoppingCriteria(eos_sequence)] if eos_sequence else None,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            prompt_len = input_data['input_ids'].shape[-1]
            generated = outputs.sequences[:, prompt_len:]
            # print("#############################")
            
            for seq in outputs.sequences:
                print(tokenizer.decode(seq, skip_special_tokens=False))
            pad_len = args.max_new_tokens - generated.size(1)
            pad_tensor = torch.full((generated.size(0), pad_len), processor.tokenizer.eos_token_id,
                                    device=generated.device, dtype=generated.dtype)
            generated_padded = torch.cat([generated, pad_tensor], dim=1)
            item_ids = encode_strings([item_id for item_id in data['item_ids'] for _ in range(args.num_return_sequences)])
            item_ids_tensor = torch.tensor(item_ids, dtype=torch.int32).to(accelerator.device)
            first_scores = outputs.scores[0]
            indices = [processor.tokenizer.convert_tokens_to_ids(t) for t in policy_tokens]
            indices = [idx for idx in indices if idx is not None and idx >= 0]
            if len(indices) == 0:
                score = torch.zeros((first_scores.shape[0], 0), device=first_scores.device, dtype=first_scores.dtype)
            else:
                score = first_scores[:, indices].softmax(dim=-1)
            all_generated_gpu.append(generated_padded)
            all_item_ids_gpu.append(item_ids_tensor)
            all_scores_gpu.append(score)
            batch_count += 1
            if batch_count % args.gather_freq == 0:
                gather_move_clear(
                    [all_generated_gpu, all_item_ids_gpu, all_scores_gpu],
                    [all_generated_cpu, all_item_ids_cpu, all_scores_cpu]
                )
            # except Exception as e:
            #     print("Error on item_ids:", encode_strings(data['item_ids']))
            #     break
            
    # 处理剩余不足batch
    if all_generated_gpu:
        gather_move_clear(
            [all_generated_gpu, all_item_ids_gpu, all_scores_gpu],
            [all_generated_cpu, all_item_ids_cpu, all_scores_cpu]
        )
    # 合并所有CPU数据
    all_generated = torch.cat(all_generated_cpu, dim=0)
    all_item_ids_tensor = torch.cat(all_item_ids_cpu, dim=0)
    all_scores = torch.cat(all_scores_cpu, dim=0)

    # decode部分保持不变
    decode_item_ids, decode_texts, decode_scores = [], [], []
    for item_ids, texts, scores in zip(all_item_ids_tensor, all_generated, all_scores):
        try:
            decode_item_ids.append(item_ids.tolist())
            decode_texts.append(
                processor.batch_decode(
                    texts.unsqueeze(0),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=True
                )[0]
            )
            decode_scores.append(scores.tolist())
        except OverflowError:
            print("OverflowError during decoding for sample:", texts[:1])
            continue

    if global_rank == 0:
        with open(args.answer_path, "w") as f1:
            for item_id, decode_text, scores in zip(decode_item_ids, decode_texts, decode_scores):
                line = {
                    'item_id': decode_string(item_id),
                    'response': decode_text,
                    'score': scores
                }
                f1.write(json.dumps(line, ensure_ascii=False) + "\n")
                f1.flush()

if __name__ == '__main__':
    # python3 vegeta_infer.py --saved_model=""
    parser = argparse.ArgumentParser(description="Run VLM model inference.")
    parser.add_argument("--saved_model", type=str,
                        default='',
                        help="Path to the saved model directory.")

    parser.add_argument("--model_version", type=str, default="mistral", choices=["mistral", "qwen", "falcon"],
                        help="the version of Vegeta Models to use. Select from ['mistral', 'qwen']")

    parser.add_argument("--question_path", type=str,
                        default='',
                        help="Path to the dataset file.")

    parser.add_argument("--answer_path", type=str,
                        default='debug/',
                        help="Filename of to model result file.")

    parser.add_argument("--max_frames", type=int, default=64,
                        help="Maximum number of sampled frames for each video.")

    parser.add_argument("--video_max_pixels", type=int, default=254016,
                        help="Maximum number of pixels for each frame in the video.")

    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference.")

    parser.add_argument("--gather_freq", type=int, default=100,
                        help="sync frequency between different GPUs.")

    parser.add_argument("--num_workers", type=int, default=16,
                        help="loader num_workers.")

    parser.add_argument("--max_new_tokens", type=int, default=16,
                        help="max new tokens for LLM output.")
    
    parser.add_argument("--num_return_sequences", type=int, default=1,
                        help="num_return_sequence")
    
    parser.add_argument("--do_sample", type=str_to_bool, default=False,
                        help="do_sample.")
    
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature.")
    
    parser.add_argument("--generation_mode", type=str_to_bool, default=False,
                        help="whether classification or generation.")        

    parser.add_argument("--policy_token_version", type=str, default='',
                        help="policy token version")        

    args = parser.parse_args()

    main(args)
import os
import json
import re
import random
import copy
from collections import defaultdict
from argparse import ArgumentParser
from pipeline.data_maker import BaseDataset

policy_definitions = '''
normal:
Typical household activities, routine pet behavior, expected daily interactions, authorized access.

abnormal:
Security threats, suspicious behavior, unauthorized access, aggressive actions, or potential safety concerns.

Priority rule: If multiple labels seem possible, choose abnormal over normal. Only output one final label.
'''

PROMPT = """
**Role:** You are an expert meta-reviewer for content safety policies. You will be given consecutive frames taken from a video.

**Task:** Evaluate whether the human reviewer's conclusion is correct, based on the policy and the provided video frames.

**Context:**
1. **Policy Definition:** {policy_rules}
2. **Reviewer's Analysis:**
   - **Conclusion:** {answer}
   - **Rationale:** {thinking}

**Output (STRICT):**
Return EXACTLY ONE of the following labels, with no extra words, punctuation, or explanation:
normal
abnormal
"""

PROMPT_POLICYTOKEN = """
**Role:** You are an expert meta-reviewer for content safety policies. You will be given consecutive frames taken from a video.

**Task:** Review the policy and the initial reviewer's analysis. Make a final, independent decision on which category the video belongs to.

**Context:**
1. **Policy Definition:** {policy_rules}
2. **Reviewer's Analysis:**
   - **Conclusion:** {answer}
   - **Rationale:** {thinking}

**Output (STRICT):**
Return EXACTLY ONE of the following labels, with no extra words, punctuation, or explanation:
normal
abnormal
"""

PROMPT_POLICYTOKEN_WITH_ASR = """
**Role:** You are an expert meta-reviewer for content safety policies. You will be given consecutive frames taken from a video.

**Task:** Review the policy and the initial reviewer's analysis. Using all available information (frames + metadata), make a final, independent decision.

**Context:**
1. **Video Information:**
   - **Title:** {title_text}
   - **ASR Transcript:** {asr_text}
   - **Sticker Text:** {sticker_text}
2. **Policy Definition:** {policy_rules}
3. **Reviewer's Analysis:**
   - **Conclusion:** {answer}
   - **Rationale:** {thinking}

**Output (STRICT):**
Return EXACTLY ONE of the following labels, with no extra words, punctuation, or explanation:
normal
abnormal
"""

PROMPT_POLICYTOKEN_ASR_WITH_LEVEL = """
**Role:** You are an expert meta-reviewer for content safety policies. You will be given consecutive frames taken from a video.

**Task:** Review the policy and the initial reviewer's analysis. Make a final, independent decision on which category the video belongs to.

**Context:**
1. **Video Information:**
   - **Title:** {title_text}
   - **ASR Transcript:** {asr_text}
   - **Sticker Text:** {sticker_text}
2. **Policy Definition:** {policy_rules}
3. **Reviewer's Analysis:**
   - **Conclusion:** {answer}
   - **Rationale:** {thinking}

**Output (STRICT):**
Return EXACTLY ONE of the following labels, with no extra words, punctuation, or explanation:
normal
abnormal
"""

def generate_pr_cot_data(args):
    policy_title_path = os.path.join(os.path.dirname(args.eval_file), "policy_info.json")
    print(args.eval_file)
    print(policy_title_path)
    print(args.mode)
    print(args.policy_book_version)
    ds_ = BaseDataset(policy_title_path)
    data = []
    with open(args.answer_file) as f:
        for line in f:
            data.append(json.loads(line))

    with open(args.eval_file) as f:
        meta_info = json.load(f)

    meta_info = {str(item['item_id']): item for item in meta_info}
    results = []
    for idx, item in enumerate(data):
        item['item_id'] = str(item['item_id'])
        if not meta_info.get(item['item_id']):
            continue

        response = item.get("response", "")

        # Try normal parsing
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)

        if answer_match and think_match:
            pred_answer = answer_match.group(1).strip()
            think = think_match.group(1).strip()
        else:
            # Fallback: put everything into think, empty answer
            pred_answer = ""
            think = response.strip()

        # continue as usual
        if 'policy_title' in meta_info[item['item_id']]:
            if args.mode == 'standard':
                policy_name = meta_info[item['item_id']]['policy_title']
                prompt = PROMPT.format(
                    policy_name=policy_name,
                    policy_rules=policy_definitions,
                    answer=pred_answer,
                    thinking=think
                )

            elif args.mode == 'policy_token':
                policy_name = meta_info[item['item_id']]['policy_title']
                policy_token = ds_.policy_title_2_policy_token[policy_name]
                prompt = PROMPT_POLICYTOKEN.format(
                    policy_name=policy_name,
                    policy_rules=policy_definitions,
                    answer=pred_answer,
                    thinking=think,
                    policy_token=policy_token
                )

            elif args.mode == 'policy_token_with_asr':
                policy_name = meta_info[item['item_id']]['policy_title']
                policy_token = ds_.policy_title_2_policy_token[policy_name]
                asr_text = meta_info[item['item_id']]['asr_text']
                title = meta_info[item['item_id']]['title']
                sticker = meta_info[item['item_id']]['sticker']
                prompt = PROMPT_POLICYTOKEN_WITH_ASR.format(
                    policy_name=policy_name,
                    title_text=title,
                    asr_text=asr_text,
                    sticker_text=sticker,
                    policy_rules=policy_definitions,
                    answer=pred_answer,
                    thinking=think,
                    policy_token=policy_token
                )

            elif args.mode == 'policy_token_asr_with_level':
                policy_name = meta_info[item['item_id']]['policy_title']
                policy_token = ds_.policy_title_2_policy_token[policy_name]
                asr_text = meta_info[item['item_id']]['asr_text']
                title = meta_info[item['item_id']]['title']
                sticker = meta_info[item['item_id']]['sticker']
                prompt = PROMPT_POLICYTOKEN_ASR_WITH_LEVEL.format(
                    policy_name=policy_name,
                    title_text=title,
                    asr_text=asr_text,
                    sticker_text=sticker,
                    policy_rules=policy_definitions,
                    answer=pred_answer,
                    thinking=think,
                    policy_token=policy_token
                )

        sample = copy.deepcopy(meta_info[item['item_id']])
        sample['conversations'][0]['content'][1]['text'] = prompt
        policy_token = ds_.policy_title_2_policy_token[meta_info[item['item_id']]['reward_label']]
        sample['conversations'][1]['content'] = policy_token #if int(meta_info[item['item_id']]['anno_label']) else ''
        results.append(sample)
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Stage2 input saved to {args.output_file}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--answer_file", type=str, default='')
    parser.add_argument("--eval_file", type=str, default='')
    parser.add_argument("--output_file", type=str, default='')
    parser.add_argument("--mode", type=str, default='policy_token_with_asr')
    parser.add_argument("--policy_book_version", type=str, default='20251105')

    args = parser.parse_args()
    generate_pr_cot_data(args)

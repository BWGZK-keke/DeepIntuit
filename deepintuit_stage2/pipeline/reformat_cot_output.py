import json

import json
import re
import random
import os
from argparse import ArgumentParser
from pipeline.data_maker import BaseDataset

def reformat(args):
    cot_ans = []
    with open(args.input_file) as f:
        for line in f:
            cot_ans.append(json.loads(line))
    with open(args.eval_file) as f:
        meta_data = json.load(f)
    meta_infos = {item['item_id']: item for item in meta_data}

    policy_title_path = os.path.join(os.path.dirname(args.eval_file), "policy_info.json")
    base_dataset = BaseDataset(policy_title_path)
    num_policies = len(base_dataset.policy_title_2_policy_token)
    
    reformat_data = []
    for item in cot_ans:
        meta_info = meta_infos[item['item_id']]
        match = re.search(r'<answer>(.*?)</answer>', item['response'], re.DOTALL)
        if not match:
            # print(item['response'])
            continue
        pred_answer = match.group(1).strip()
        if 'policy_title' in meta_info and 'Bullying' in meta_info['policy_title']:
            if 'no' in pred_answer.lower():
                label = 0
                policy_name = meta_info['policy_title']
                token_idx = base_dataset.policy_title_2_policy_idx[policy_name]
            elif 'moderate' in pred_answer.lower():
                label = 1
                token_idx = base_dataset.policy_title_2_policy_idx['Moderate Bullying']
            else:
                label = 1
                token_idx = base_dataset.policy_title_2_policy_idx['Severe Bullying']
        else:
            if 'yes' in pred_answer.lower():
                label = 1
            else:
                label = 0
            
            if 'policy_title' in meta_info:
                policy_name = meta_info['policy_title']
            else:
                policy_name = 'Personal Information - High Risk'

            token_idx = base_dataset.policy_title_2_policy_idx[policy_name]
        scores = [0] * (num_policies + 1)
        if label == 1:
            scores[0] = 0.0
            scores[token_idx] = 1.0
        else:
            scores[0] = 1.0
            scores[token_idx] = 0.0
        item['score'] = scores
        reformat_data.append(item)
    
    with open(args.output_file, 'w') as f1:
        for line in reformat_data:
            f1.write(json.dumps(line, ensure_ascii=False) + "\n")
            f1.flush()
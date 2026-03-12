import json

import json
import re
import random
import os
from argparse import ArgumentParser
from pipeline.data_maker import BaseDataset

def reformat(args):
    data = []
    with open(args.input_file) as f:
        for line in f:
            data.append(json.loads(line))

    with open(args.eval_file) as f:
        meta_data = json.load(f)
    meta_infos = {item['item_id']: item for item in meta_data}

    cot_ans = []
    with open(args.question_file) as f:
        for line in f:
            cot_ans.append(json.loads(line))
    print(args.question_file)
    cot_ans = {item['item_id']: item for item in cot_ans}
    policy_title_path = os.path.join(os.path.dirname(args.eval_file), "policy_info.json")
    base_dataset = BaseDataset(policy_title_path)
    num_policies = len(base_dataset.policy_title_2_policy_token)
    
    reformat_data = []
    for item in data:
        meta_info = meta_infos[item['item_id']]
        cot = cot_ans[item['item_id']]
        match = re.search(r'<answer>(.*?)</answer>', cot['response'], re.DOTALL)
        if not match:
            print(cot['response'])
            continue
        pred_answer = match.group(1).strip()
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
            scores[0] = item['score'][0]
            scores[token_idx] = item['score'][1]
        else:
            scores[0] = item['score'][1]
            scores[token_idx] = item['score'][0]    
        item['score'] = scores
        reformat_data.append(item)
    
    with open(args.output_file, 'w') as f1:
        for line in reformat_data:
            f1.write(json.dumps(line, ensure_ascii=False) + "\n")
            f1.flush()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, default='/opt/tiger/themis/debug/pr_eval_stage1_subset_cot_stage2_subset_cot_model.json')
    parser.add_argument("--output_file", type=str, default='')
    parser.add_argument("--eval_file", type=str, default='/mnt/bn/themis/data/themis_workspace/xiangchen.zhao/20250525/personal_risk_0720_PA_eval_debug.json')
    parser.add_argument("--question_file", type=str, default='/opt/tiger/themis/debug/pr_eval_cot.json')

    args = parser.parse_args()
    reformat(args)


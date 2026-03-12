import re
illegal_word_list = ['<unk>', '<s>', '</s>', '[INST]', '[/INST]', '<vis_start>', '<vis_end>', \
                     '<img_pad>', '<vid_pad>', '<think>', '</think>', '<answer>', '</answer>']



def violation_compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    solution_str = solution_str.replace('<unk>', '')
    format_score = format_reward(solution_str)
    accuracy_score = accuracy_reward(solution_str, ground_truth)
    return {
        "score": accuracy_score + format_score,
        "format": format_score,
        "accuracy": accuracy_score,        
    }

def violation_bullying_compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    solution_str = solution_str.replace('<unk>', '')
    format_score = format_reward(solution_str)
    accuracy_score_bullying = accuracy_reward_bullying(solution_str, ground_truth)
    return {
        "score": accuracy_score_bullying + format_score,
        "format": format_score,
        "accuracy": accuracy_score_bullying,        
    }

def violation_level_compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    solution_str = solution_str.replace('<unk>', '')
    format_score = format_reward(solution_str)
    accuracy_score = accuracy_level_reward(solution_str, ground_truth)
    return {
        "score": accuracy_score + format_score,
        "format": format_score,
        "accuracy": accuracy_score,        
    }

def accuracy_level_reward(completion, sol):
    """Reward function that checks if the completion is the same as the ground truth."""
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', completion)
    gap = 3.0
    try:
        content = match.group(1).strip()
        gap = abs(int(content) - int(sol))
    except:
        pass
    if gap == 0.0:
        reward = 1.0
    elif gap > 2.0:
        reward = 0.0
    else:
        reward = 1/(gap+1)
    return reward



def accuracy_reward(completion, sol):
    """Reward function that checks if the completion is the same as the ground truth."""
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', completion)
    pred_reward_label = -1
    try:
        content = match.group(1).strip()
        if content.lower() == 'no':
            pred_reward_label = 0
        elif content.lower() == 'yes':
            pred_reward_label = 1
        else:
            pred_reward_label = -1
    except:
        pass
    if pred_reward_label == -1:
        reward = 0.0  # can't parsing, wrong answer
    elif pred_reward_label == sol:
        reward = 1.0
    else:
        reward = 0.0
    return reward


def accuracy_reward_bullying(completion, sol):
    """Reward function that checks if the completion is the same as the ground truth."""
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', completion)
    corrected = False
    try:
        content = match.group(1).strip()
        corrected = (content.lower() == sol.strip().lower())
    except:
        pass
    if corrected:
        reward = 1.0
    else:
        reward = 0.0
    return reward


def format_reward(completion):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>(?:\s*</s>)?\s*$"
    match = re.match(pattern, completion, re.DOTALL)
    if match:
        pred_answer = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL).group(1).strip()
        pred_thinking = re.search(r'<think>(.*?)</think>', completion, re.DOTALL).group(1).strip()
        if any(illegal_word in pred_answer for illegal_word in illegal_word_list) or any(illegal_word in pred_thinking for illegal_word in illegal_word_list):
            reward = 0
        else:
            reward = 1
    else:
        reward = 0
    return reward

def accuracy_reward_with_rule(completion, sol):
    """Reward function that checks if the completion is the same as the ground truth."""
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', completion)
    answer_gt, tree_gt = sol
    pred_reward_label = -1
    try:
        content = match.group(1).strip()
        if content.lower().startswith('no'):
            pred_reward_label = 0
        elif content.lower().startswith('yes'):
            pred_reward_label = 1
        else:
            pred_reward_label = -1
    except:
        pass
    if pred_reward_label == -1:
        reward = 0.0  # can't parsing, wrong answer
    elif pred_reward_label == answer_gt:
        reward = 1.0
    else:
        reward = 0.0
    
    tree_reward = 0
    match = re.search(r'<HIT>\s*(.*?)\s*</HIT>', completion)
    try:
        content = match.group(1).strip()

        if tree_gt == -1:
            if 'none' in content.lower():
                tree_reward = 1.0
        else:
            pattern = r'\d+'
            num_matches = re.findall(pattern, content)
            num_matches = sorted([int(num) for num in num_matches])
            if len(num_matches) > 0 and num_matches[0] == tree_gt:
                    tree_reward = 1.0
    except:
        pass         

    return reward + tree_reward



def format_reward_with_rule(completion):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*<HIT>.*?</HIT>\s*$"
    match = re.match(pattern, completion, re.DOTALL)
    return 1.0 if match else 0.0

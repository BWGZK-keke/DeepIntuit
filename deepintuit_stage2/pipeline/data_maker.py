"""
Your code is your poem, your comments are its interpretation.
"""
import json
import logging
import os
import random

from tqdm.auto import tqdm

logger = logging.getLogger(os.path.basename(__file__))


class BaseDataset:
    def __init__(self, policy_info_path=None):
        print("#########################")
        print(policy_info_path)

   
        self.POLICY_INFO_PATH = policy_info_path

        if os.path.exists(self.POLICY_INFO_PATH):
            self.policy_info = json.load(open(self.POLICY_INFO_PATH))

        self.policy_info_sorted = sorted(list(self.policy_info.items()))
        self.all_policy_info_list = [
            [line[0], line[1][0]['Policy Code'], line[1][0]['Policy Token']] for line in self.policy_info_sorted]

        self.policy_title_2_policy_token, self.policy_title_2_policy_code = {}, {}
        self.policy_code_2_policy_token, self.policy_code_2_policy_title = {}, {}
        self.policy_title_2_policy_idx, self.policy_idx_2_policy_title = {}, {}
        self.policy_token_2_policy_title, self.policy_token_2_policy_idx = {}, {}
        self.policy_idx_2_policy_token = {}

        for i, (policy_title, policy_code, policy_token) in enumerate(self.all_policy_info_list):
            self.policy_title_2_policy_idx[policy_title] = i + 1
            self.policy_idx_2_policy_title[i + 1] = policy_title
            self.policy_idx_2_policy_token[i + 1] = policy_token
            self.policy_title_2_policy_code[policy_title] = policy_code
            self.policy_title_2_policy_token[policy_title] = policy_token
            self.policy_token_2_policy_title[policy_token] = policy_title
            self.policy_token_2_policy_idx[policy_token] = i + 1

            for code in policy_code:
                self.policy_code_2_policy_token[code] = policy_token
                self.policy_code_2_policy_title[code] = policy_title

        # 真正使用的 prompt
        self.global_prompt = self.generate_global_prompt(self.all_policy_info_list)
        self.global_short_prompt = self.generate_global_short_prompt()

    @staticmethod
    def generate_global_prompt(policy_info_list):
        policy_choice_string = f"Your job is to determine if this video falls under one of the 3 categories: Normal_Content, Hateful_Content, and Offensive_Content."
        last_string = f"Your answer should be on word of Normal_Content, Hateful_Content, or Offensive_Content."
        #for i, (policy_title, policy_code, policy_token) in enumerate(policy_info_list):
            #policy_choice_string += f", {i + 2}) Violate policy '{policy_title}': {policy_token}"
            #last_string += f", {policy_token}"

        #policy_choice_string += "."
        #last_string += "]."

        global_prompt = f'''You are now ContentModGPT, an expert in content moderation.
You will be given consecutive frames taken from a video, with a 0.5-second gap between each frame. The video footage might be shaky, unclear, or tilted. Please attempt to comprehend the video's content as much as possible and provide a response, but do not infer content that is not present in the video.
{policy_choice_string}
{last_string}

''' + '''Video text information:  {all_text}'''

        return global_prompt

    @staticmethod
    def generate_global_short_prompt():
        # I like it!
        global_short_prompt = '''You are now ContentModGPT, an expert in content moderation.
You will be given consecutive frames taken from a video, with a 0.5-second gap between each frame. The video footage might be shaky, unclear, or tilted. Please attempt to comprehend the video's content as much as possible and provide a response, but do not infer content that is not present in the video.
Your job is to determine if this video falls under one of the specific categories,i.e., Normal_Content, Hateful_Content, and Offensive_Content. :

''' + '''Video text information:  {all_text}'''

        return global_short_prompt

    @staticmethod
    def build_candidate_samples(prompt, item_id, info_dict, frame_path, caption=None):

        # 处理文本信息
        text_info = []
        for key in ["title", "sticker", "ocr_text", "asr_text", "background_asr_text"]:
            if key in info_dict and len(info_dict[key]) > 0:
                text_info.append(info_dict[key])

        all_text = "\t".join(text_info)[:1024]

        using_prompt = prompt.format(
            all_text=all_text)

        if caption is not None:
            candidate_data = {
                "item_id": item_id,
                "conversations":
                    [
                        {"role": "user", "content": [
                            {
                                "type": "video",
                                "video": frame_path
                            },
                            {
                                "type": "text",
                                "text": using_prompt
                            }
                        ]},
                        {"role": "assistant", "content": caption}
                    ]
            }
        else:
            candidate_data = {
                "item_id": item_id,
                "conversations":
                    [
                        {"role": "user", "content": [
                            {
                                "type": "video",
                                "video": frame_path
                            },
                            {
                                "type": "text",
                                "text": using_prompt
                            }
                        ]},
                    ]
            }

        for key in info_dict:
            if key.startswith("gift"):
                candidate_data[key] = info_dict[key]

        return candidate_data


class VideoProcessor(BaseDataset):
    def __init__(self, meta_info, data_config, base_config):
        super(VideoProcessor, self).__init__(policy_info_path=meta_info.policy_info_tos_path)

        self.meta_info = meta_info
        self.data_config = data_config
        self.base_config = base_config

        if self.base_config.get("overwrite_raw_data", False):
            logger.warning(f"overwrite_raw_data is Open, will reproduce raw data(like rerun SQL)...")

        if self.base_config.get("overwrite_task_data", False):
            logger.warning(f"overwrite_task_data is Open, will overwrite model train file...")

        if self.base_config.get("only_use_frame_cache", False):
            logger.warning(f"only_use_frame_cache is Open, only using cached frames, will be fast...")

        # 相关路径
        self.hdfs_root_path = (f'{self.meta_info.hdfs_path}/{self.meta_info.owner}/'
                               f'{self.meta_info.job_name}/{self.meta_info.date}/raw_{self.data_config.name}')

        self.raw_data_path = f'{self.base_config.job_base_path}/raw_{self.data_config.name}.json'
        self.task_data_path = f"{self.base_config.job_base_path}/{self.data_config.name}.json"

        # 信息备份
        with open(f"{self.base_config.job_base_path}/policy_info.json", 'w') as f1:
            json.dump(self.policy_info, f1, indent=4, ensure_ascii=False)

    def build_task_dataset(self, raw_data_with_frame):
        # 分别构造正负样本数据
        title_2_pos_data_dict = {title: [] for title in self.policy_title_2_policy_code}
        pos_data_list, neg_data_list = [], []

        for item_id, info_dict, frame_path in tqdm(raw_data_with_frame):
            true_policy_token = [
                [self.policy_code_2_policy_title[code], self.policy_code_2_policy_token[code]]
                for code in set(info_dict.get("true_policy_code_list", [])) if code in self.policy_code_2_policy_token]

            tcs_policy_token = [
                [self.policy_code_2_policy_title[code], self.policy_code_2_policy_token[code]]
                for code in set(info_dict.get("origin_policy_code_list", [])) if
                code in self.policy_code_2_policy_token]

            title_token_sample = [
                [title, self.build_candidate_samples(self.global_short_prompt, item_id, info_dict, frame_path, token)]
                for title, token in true_policy_token]

            for title, token_sample in title_token_sample:
                token_sample.update(
                    {
                        "true_title": [title for title, token in true_policy_token],
                        "origin_title": [title for title, token in tcs_policy_token],
                        "true_token": [token for title, token in true_policy_token],
                        "origin_token": [token for title, token in tcs_policy_token],
                    }
                )
                title_2_pos_data_dict[title].append(token_sample)

            if len(title_token_sample) == 0:
                token_sample = self.build_candidate_samples(
                    self.global_short_prompt, item_id, info_dict, frame_path, "<NO_POLICY>")
                token_sample.update(
                    {
                        "true_title": [title for title, token in true_policy_token],
                        "origin_title": [title for title, token in tcs_policy_token],
                        "true_token": [token for title, token in true_policy_token],
                        "origin_token": [token for title, token in tcs_policy_token],
                    }
                )
                neg_data_list.append(token_sample)

        # 混合正负样本训练数据
        for title in title_2_pos_data_dict:
            pos_data = title_2_pos_data_dict[title][-self.data_config.get('max_pos_number_pre_policy', 9999999):]
            pos_data_list.extend(pos_data)
            logger.warning(f"Task Data --> "
                           f"title: {title}, "
                           f"pos_number: {len(pos_data)}")

        # train_pos + train_neg --> train
        all_token_list = pos_data_list + neg_data_list

        logger.warning(f"\nall {self.data_config.name} count: {len(all_token_list)}, "
                       f"\npos {self.data_config.name} count: {len(pos_data_list)}, "
                       f"\nneg {self.data_config.name} count: {len(neg_data_list)}")

        random.shuffle(all_token_list)
        with open(self.task_data_path, "w") as f1:
            json.dump(all_token_list, f1, indent=4, ensure_ascii=False)

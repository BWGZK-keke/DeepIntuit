import io
import json
import os
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from typing import List

import bytedtos
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers.trainer_pt_utils import LabelSmoother

def read_zip_as_tos(zip_path):
    """
    Reads a local zip file and mimics TOS get_object output.
    
    Returns a simple object with a `.data` attribute containing the zip bytes.
    """
    class LocalTOSObject:
        def __init__(self, data: bytes):
            self.data = data

    with open(zip_path, "rb") as f:
        data = f.read()

    return LocalTOSObject(data)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class NovaDataset(Dataset):
    def __init__(self, file_path, max_frames=32, train_mode=True):
        self.max_frames = max_frames
        # create u13 filter only when actively enabled and during training
        u13_enabled = is_u13_filter_enabled() and train_mode
        if u13_enabled:
            u13_cfg_mgr, u13_filter = create_single_u13_filter([file_path], 'item_id')

        self.data = []
        data_before_filter = json.load(open(file_path))
        row_count_total = len(data_before_filter)
        # per row filter to save memory usage
        for row in data_before_filter:
            # do the filter
            if not u13_enabled:
                self.data.append(row)
                continue
            filtered_row = u13_filter.filter([row])
            if filtered_row is not None:
                self.data.append(filtered_row[0])

        if u13_enabled:
            u13_filter.report_metrics()
            print(f"before applying u13 filter, we have {row_count_total} data")
            print(f"after applying u13 filter, we have {len(self.data)} data")
            # clean up u13 filter related objects
            del u13_filter.u13_cache
            del u13_filter
            del u13_cfg_mgr

        # convert themis data format into vegeta format
        if "image" in self.data[0].keys():
            print("converting themis data format into vegeta format")
            for row in tqdm(self.data):
                if train_mode:
                    row["conversations"] = [
                        {"role": "user", "content": [
                            {
                                "type": "video",
                                "video": row["image"]
                            },
                            {
                                "type": "text",
                                "text": row["conversations"][0]["value"].replace("<image>", "").lstrip()
                            }
                        ]},
                        {"role": "assistant", "content": row["conversations"][1]["value"]}
                    ]
                else:
                    # during inference, we drop the assistant response
                    row["conversations"] = [
                        {"role": "user", "content": [
                            {
                                "type": "video",
                                "video": row["image"]
                            },
                            {
                                "type": "text",
                                "text": row["conversations"][0]["value"].replace("<image>", "").lstrip()
                            }
                        ]}
                    ]

    def get_from_tos(self, filepath):
        # the start_index corresponds to the local tos
        resp = read_zip_as_tos(filepath)
        raw_data = io.BytesIO(resp.data)
        images = []
        with zipfile.ZipFile(raw_data, 'r') as zip_file:
            for zip_image in zip_file.namelist():
                with zip_file.open(zip_image) as image_data:
                    image = Image.open(io.BytesIO(image_data.read()))
                    images.append(image.convert('RGB'))
        return images

    def __len__(self):
        return len(self.data)

    def load_image(self, filepath):
        if not os.path.exists(filepath):
            return None
        try:
            return Image.open(filepath)
        except Exception as e:
            print(f"inside load_image, error : {e}")
            return None

    def uniform_sample(self, frames, num_segments):
        """
        Uniformly samples 10 frames from a list of frames.

        Args:
        - frames (list): A list of frames.

        Returns:
        - list: A list containing 10 uniformly sampled frames.
        """

        indices = np.linspace(start=0, stop=len(frames) - 1, num=num_segments).astype(int)

        frames = [frames[ind] for ind in indices]

        return frames

    def load_compressed_video(self, filepath):
        images = []
        try:
            # determine if the file exists on tos buckets
            if "video_frames" in filepath and filepath.endswith(".zip"):
                images = self.get_from_tos(filepath)
            # if the video path is a normal zip file path, we read it from local storage
            elif isinstance(filepath, str) and filepath.startswith("/") and filepath.endswith(".zip"):
                with zipfile.ZipFile(filepath, 'r') as zip_file:
                    for zip_image in zip_file.namelist():
                        with zip_file.open(zip_image) as image_data:
                            image = Image.open(io.BytesIO(image_data.read()))
                            images.append(image.convert('RGB'))
            else:
                # compatible with old themis data storage methods(nas list)
                for image_file in filepath:
                    image = self.load_image(image_file)
                    if not image:
                        return None
                    images.append(image.convert('RGB'))

            if len(images) < 2:
                print(f"Compressed package {filepath} contains less than 2 valid images")
                # if there's only one image, we duplicate it to make it compatible with 3D convolution
                if len(images) == 1:
                    return images + images
                return None

            # Perform sampling and adjust to even number of images
            if len(images) > self.max_frames:
                images = self.uniform_sample(images, self.max_frames)
            elif len(images) % 2 != 0:
                images = images[:-1]  # Adjust to even number of frames

            return images

        except Exception as e:
            print(f"Error loading compressed video: {e}, file path: {filepath}")
            return None

    def __getitem__(self, idx):
        while True:
            item = deepcopy(self.data[idx])

            images, videos = [], []
            sample_error = False

            # Iterate through conversation contents
            for message in item['conversations']:
                if not isinstance(message['content'], list):
                    continue

                for content in message['content']:
                    if content.get('type') == 'image' or 'image' in content:
                        # Try to load the image
                        image = self.load_image(content.get('image', ''))
                        if image:
                            images.append(image)
                        else:
                            sample_error = True  # Image loading failed
                            break
                    elif content.get('type') == 'video' or 'video' in content:
                        # Try to load the compressed video
                        video = self.load_compressed_video(content.get('video', ''))
                        if video:
                            videos.append(video)
                        else:
                            sample_error = True  # Video loading failed
                            break

                if sample_error:
                    break  # Exit the current record processing logic

            # If no errors, return the result
            if not sample_error:
                item.update({'images': images, 'videos': videos})
                return item

            # If there is an error, randomly fetch another idx
            idx = np.random.randint(0, len(self.data))


@dataclass
class NovaCollator:
    data_type: str = 'bf16'
    processor: object = None
    train_mode: bool = True
    assistant_prompt: str = "<|im_start|>assistant\n"
    eos_token: str = "<|im_end|>\n"

    def __post_init__(self):
        self.assistant_token_ids = self.processor.tokenizer(self.assistant_prompt)['input_ids']
        self.eos_ids = self.processor.tokenizer(self.eos_token)["input_ids"]

    @staticmethod
    def find_subsequence_in_batch(batch_tensor, sub_tensor1, sub_tensor2):
        """
        在 batch_tensor 中找到以 sub_tensor1 开头、以 sub_tensor2 结束的所有子序列的起止位置。

        :param batch_tensor: 主张量 (形状: [batch_size, seq_len], torch.Tensor)
        :param sub_tensor1: 起始子张量 (list 类型)
        :param sub_tensor2: 结束子张量 (list 类型)
        :return: 两个元组，分别表示起始位置和结束位置的 (row_indices, col_indices)
        """
        # 将 list 转换为 torch.Tensor
        sub_tensor1 = torch.tensor(sub_tensor1, dtype=batch_tensor.dtype, device=batch_tensor.device)
        sub_tensor2 = torch.tensor(sub_tensor2, dtype=batch_tensor.dtype, device=batch_tensor.device)

        batch_size, seq_len = batch_tensor.size()
        sub_len1 = sub_tensor1.size(0)
        sub_len2 = sub_tensor2.size(0)

        if sub_len1 > seq_len or sub_len2 > seq_len:
            raise ValueError("Subsequence should not be longer than main sequence.")

        # 找到 sub_tensor1 的所有起始位置
        sliding_window1 = batch_tensor.unfold(dimension=1, size=sub_len1, step=1)
        matches1 = (sliding_window1 == sub_tensor1.unsqueeze(0).unsqueeze(0)).all(dim=-1)
        start_indices = torch.nonzero(matches1, as_tuple=False)

        # 找到 sub_tensor2 的所有起始位置
        sliding_window2 = batch_tensor.unfold(dimension=1, size=sub_len2, step=1)
        matches2 = (sliding_window2 == sub_tensor2.unsqueeze(0).unsqueeze(0)).all(dim=-1)
        end_indices = torch.nonzero(matches2, as_tuple=False)

        # 过滤 start 和 end 的匹配，使得 end_index > start_index + len(sub_tensor1)
        start_row_indices = []
        start_col_indices = []
        end_row_indices = []
        end_col_indices = []

        for start in start_indices:
            batch_idx, start_idx = start.tolist()
            for end in end_indices:
                end_batch_idx, end_idx = end.tolist()
                if batch_idx == end_batch_idx and end_idx > start_idx + sub_len1 - 1:
                    start_row_indices.append(batch_idx)
                    start_col_indices.append(start_idx)
                    end_row_indices.append(end_batch_idx)
                    end_col_indices.append(end_idx)
                    break  # 每个起始点只匹配第一个符合条件的结束点

        return (
            (torch.tensor(start_row_indices, device=batch_tensor.device),
             torch.tensor(start_col_indices, device=batch_tensor.device)),
            (torch.tensor(end_row_indices, device=batch_tensor.device),
             torch.tensor(end_col_indices, device=batch_tensor.device))
        )

    def __call__(self, features):
        item_ids, images, videos, texts = [], [], [], []

        for feature in features:
            images.extend(feature['images'])
            videos.extend(feature['videos'])
            if self.train_mode:
                texts.append(self.processor.apply_chat_template(feature['conversations']))
            else:
                # texts.append(self.processor.apply_chat_template(feature['conversations'][:1], add_generation_prompt=True))
                texts.append(self.processor.apply_chat_template(feature['conversations'], add_generation_prompt=True))
                item_ids.append(str(feature['item_id']))

        if not images:
            images = None
        if not videos:
            videos = None

        batch_dict = self.processor(text=texts, images=images, videos=videos, padding=True, return_tensors='pt')

        # labels = torch.clone(batch_dict['input_ids'])
        if self.train_mode:
            input_ids = batch_dict['input_ids']
            labels = torch.ones_like(input_ids, dtype=input_ids.dtype) * IGNORE_TOKEN_ID

            start_ids, end_ids = self.find_subsequence_in_batch(input_ids, self.assistant_token_ids, self.eos_ids)

            assert (len(start_ids[0]) == len(end_ids[0])), "Number of input start and end tokens not match."
            for i, start, end in zip(start_ids[0], start_ids[1], end_ids[1]):
                assert start < end, f"Start pos {start} must be ealier than end pos {end}."
                labels[i, start + len(self.assistant_token_ids): end + len(self.eos_ids)] = input_ids[i, start + len(
                    self.assistant_token_ids): end + len(self.eos_ids)]

            batch_dict.update({'labels': labels})
        else:
            batch_dict.update({'item_ids': item_ids})

        for k, v in batch_dict.items():
            if type(v) is list:
                continue
            if self.data_type == 'bf16' and v.dtype == torch.float:
                batch_dict[k] = v.to(torch.bfloat16)
            elif self.data_type == 'fp16' and v.dtype == torch.float:
                batch_dict[k] = v.to(torch.float16)

        return batch_dict


def create_single_u13_filter(sources: List[str], u13_id_field: str):
    """
    create the filter based on item id
    """
    from bytedance.data_compliance.u13 import U13Config
    from bytedance.data_compliance.u13.u13_filter import U13_ITEM_ID
    u13_cfg_mgr = U13Config(
        data_sources=sources,
        data_formats=['jsonl'],  # source_types in {'parquet', 'jsonl', 'csv'}
        u13_id_fields=[u13_id_field],  # cruise will load the enviromental variable, no need to set here
        u13_id_labels=[U13_ITEM_ID]
    )
    filters = u13_cfg_mgr.get_filters()
    if not filters or sources[0] not in filters:
        raise Exception('create u13 filter failed')
    u13_filter = filters[sources[0]]
    return u13_cfg_mgr, u13_filter


def is_u13_filter_enabled() -> bool:
    return os.getenv('USE_U13_FILTER') in ['TRUE', 'true', 'True', '1']

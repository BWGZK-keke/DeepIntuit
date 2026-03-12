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
import zipfile
import json
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
    def __init__(self, file_path, max_frames=32, train_mode=False):
        self.max_frames = max_frames
        self.train_mode = train_mode

        # create u13 filter only when actively enabled and during training
        u13_enabled = is_u13_filter_enabled() and train_mode
        if u13_enabled:
            u13_cfg_mgr, u13_filter = create_single_u13_filter([file_path], 'item_id')

        self.data = None
        self.filter_file_path = None
        if file_path.endswith('jsonl'):
            self.filter_file_path = file_path
            index_path = file_path.replace('jsonl', 'index')
            print(f"Loading index from {index_path}...")
            with open(index_path, 'r') as f:
                # 将所有字节偏移量一次性加载到内存
                self.offsets = [int(line.strip()) for line in f]
            self.data_len = len(self.offsets)
        else:
            data_before_filter = json.load(open(file_path, 'r', encoding='utf-8'))

            if not self.train_mode:
                # 推理状态下，仅保留问题部分
                print(f"### infer model, will filter assistant part.....")
                for line in data_before_filter:
                    line["conversations"] = line["conversations"][:1]

            if not u13_enabled:
                self.data = data_before_filter
            else:
                row_count_total = len(data_before_filter)
                # per row filter to save memory usage
                for row in data_before_filter:
                    # do the filter
                    filtered_row = u13_filter.filter([row])
                    if filtered_row is not None:
                        self.data.append(filtered_row[0])

                u13_filter.report_metrics()
                print(f"before applying u13 filter, we have {row_count_total} data")
                print(f"after applying u13 filter, we have {len(self.data)} data")
                # clean up u13 filter related objects
                del u13_filter.u13_cache
                del u13_filter
                del u13_cfg_mgr
            self.data_len = len(self.data)
       
    @staticmethod
    def get_from_tos(filepath):
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
        return self.data_len


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
            if "video_frames_file" in filepath and filepath.endswith(".zip"):
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
            # copy the item to avoid storing images and videos in memory
            if self.filter_file_path:
                if self.data is None:
                    self.data = open(self.filter_file_path, 'r', encoding='utf-8')
                offset = self.offsets[idx]
                self.data.seek(offset)
                line = self.data.readline()
                item = json.loads(line)
                if not self.train_mode:
                    item["conversations"] = item["conversations"][:1]
            else:
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
            idx = np.random.randint(0, self.data_len)


@dataclass
class NovaCollator:
    data_type: str = 'bf16'
    processor: object = None
    train_mode: bool = True
    input_start_id: int = 151644
    input_end_id: int = 151645

    def __post_init__(self):
        self._missing_assistant_prefix_total = 0
        self._missing_assistant_end_total = 0
        self._mask_warning_prints = 0

    def __call__(self, features):
        item_ids, images, videos, texts = [], [], [], []
        for feature in features:
            images.extend(feature['images'])
            videos.extend(feature['videos'])
            # images = [f.get("images", []) for f in features]   # list-of-list
            # videos = [f.get("videos", []) for f in features]  

            if self.train_mode:
                convs = deepcopy(feature["conversations"])
                if convs and convs[-1]["role"] == "assistant":
                    convs[-1]["content"] = feature["reward_label"]
                texts.append(self.processor.apply_chat_template(convs)+self.processor.tokenizer.eos_token)
            else:
                item_ids.append(str(feature['item_id']))
                convs = deepcopy(feature["conversations"])
                if convs and convs[-1]["role"] == "assistant":
                    convs[-1]["content"] = ""
                texts.append(self.processor.apply_chat_template(convs, add_generation_prompt=True))

        if not images:
            images = None
        if not videos:
            videos = None
        batch_dict = self.processor(text=texts, images=images, videos=videos, padding=True, return_tensors='pt')
        # print(texts)
        labels = torch.clone(batch_dict['input_ids'])
        input_ids = batch_dict["input_ids"]
        
        IGNORE = -100
        tok = self.processor.tokenizer

        # start by ignoring everything
        labels[:] = IGNORE

        # tokens that mark the start of assistant content in Qwen chat
        assistant_prefix_ids = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
        assistant_end_ids = tok.encode("<|im_end|>", add_special_tokens=False)
        missing_prefix_in_batch = 0
        missing_end_in_batch = 0

        for b in range(input_ids.size(0)):
            ids = input_ids[b].tolist()

            # find LAST occurrence of the assistant prefix
            start = -1
            for j in range(max(0, len(ids) - len(assistant_prefix_ids) + 1)):
                if ids[j:j + len(assistant_prefix_ids)] == assistant_prefix_ids:
                    start = j + len(assistant_prefix_ids)

            # no assistant marker => keep all ignore
            if start == -1:
                missing_prefix_in_batch += 1
                continue

            # find first <|im_end|> after assistant content starts
            end = -1
            for j in range(start, max(start, len(ids) - len(assistant_end_ids) + 1)):
                if ids[j:j + len(assistant_end_ids)] == assistant_end_ids:
                    end = j
                    break

            # missing <|im_end|> => keep all ignore for safety
            if end == -1:
                missing_end_in_batch += 1
                continue

            if end > start:
                labels[b, start:end] = input_ids[b, start:end]

        if self.train_mode and (missing_prefix_in_batch > 0 or missing_end_in_batch > 0):
            self._missing_assistant_prefix_total += missing_prefix_in_batch
            self._missing_assistant_end_total += missing_end_in_batch
            if self._mask_warning_prints < 5:
                print(
                    "[NovaCollator] assistant span parse warning: "
                    f"missing_prefix={missing_prefix_in_batch}, missing_im_end={missing_end_in_batch}, "
                    f"cumulative_missing_prefix={self._missing_assistant_prefix_total}, "
                    f"cumulative_missing_im_end={self._missing_assistant_end_total}"
                )
                self._mask_warning_prints += 1

        # ignore padding
        if tok.pad_token_id is not None:
            labels[labels == tok.pad_token_id] = IGNORE

        # optionally ignore eos / <|im_end|>
        if tok.eos_token_id is not None:
            labels[labels == tok.eos_token_id] = IGNORE

        # (optional but recommended for VL) ignore vision placeholder tokens
        for t in ["<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>"]:
            if t in tok.get_vocab():
                tid = tok.convert_tokens_to_ids(t)
                labels[labels == tid] = IGNORE

        batch_dict["labels"] = labels
        # print("##############################")
        # print(batch_dict["input_ids"])

        if not self.train_mode:
            batch_dict.update({'item_ids': item_ids})

        for k, v in batch_dict.items():
            if type(v) is list:
                continue
            if self.data_type == 'bf16' and v.dtype == torch.float:
                batch_dict[k] = v.to(torch.bfloat16)
            elif self.data_type == 'fp16' and v.dtype == torch.float:
                batch_dict[k] = v.to(torch.float16)

        return batch_dict



@dataclass
class NovaSeqClsCollator:
    data_type: str = 'bf16'
    processor: object = None
    train_mode: bool = False
    input_start_id: int = 151644
    input_end_id: int = 151645
    policy_token_2_policy_idx: dict = None
    num_labels: int = 100
    label_type: str = ''
    def __post_init__(self):
        pass

    def __call__(self, features):
        item_ids, images, videos, texts, labels_info = [], [], [], [], []
        for feature in features:
            images.extend(feature['images'])
            videos.extend(feature['videos'])

            if 'cls_label' in feature:
                labels_info.append(feature['cls_label'])
            else:
                labels_info.append(feature['conversations'][1])

            texts.append(
                self.processor.apply_chat_template(feature['conversations'][:1]) + self.processor.tokenizer.eos_token)
     
            if not self.train_mode:
                item_ids.append(str(feature['item_id']))
               
        if not images:
            images = None
        if not videos:
            videos = None

        batch_dict = self.processor(text=texts, images=images, videos=videos, padding=True, return_tensors='pt')
        labels = []
        if 'cls_label' in features[0]:
            NotImplementedError
        else:
            for label_info in labels_info:
                label_idx = self.policy_token_2_policy_idx[label_info['content']]
                if label_idx == 0: # approved
                    label = torch.zeros(self.num_labels, dtype=torch.int)
                    label[0] = 1
                else:
                    if self.label_type == 'one_hot':
                        label = torch.zeros(self.num_labels, dtype=torch.int)
                        label[label_idx] = 1
                    else:
                        label = -1 * torch.ones(self.num_labels, dtype=torch.int) 
                        label[label_idx] = 1
                        label[0] = 0
                labels.append(label)
            labels = torch.stack(labels, dim=0) if len(labels) > 0 else []
        labels = torch.clone(batch_dict['input_ids'])
        start_ids = torch.where(labels == self.input_start_id)
        end_ids = torch.where(labels == self.input_end_id)
        assert (len(start_ids[0]) == len(end_ids[0])), "Number of input start and end tokens not match."
        for i, start, end in zip(start_ids[0], start_ids[1], end_ids[1]):
            assert start < end, f"Start pos {start} must be ealier than end pos {end}."
            labels[i, start: end + 1] = IGNORE_TOKEN_ID

        pad_ids = torch.where(labels == self.processor.tokenizer.pad_token_id)
        bos_ids = torch.where(labels == self.processor.tokenizer.bos_token_id)

        labels[pad_ids] = IGNORE_TOKEN_ID
        labels[bos_ids] = IGNORE_TOKEN_ID
        batch_dict.update({'labels': labels})

        if not self.train_mode:
            batch_dict.update({'item_ids': item_ids})

        for k, v in batch_dict.items():
            if type(v) is list:
                continue
            if self.data_type == 'bf16' and v.dtype == torch.float:
                batch_dict[k] = v.to(torch.bfloat16)
            elif self.data_type == 'fp16' and v.dtype == torch.float:
                batch_dict[k] = v.to(torch.float16)

        return batch_dict
    
# @dataclass
# class NovaVllmCollator:
#     processor: object = None
#     train_mode: bool = True

#     def __call__(self, features):
#         item_ids, videos, texts, multi_modal_data = [], [], [], []

#         for feature in features:
#             videos.append(feature['videos'][0] if feature['videos'] else None)
#             if self.train_mode:
#                 texts.append(
#                     self.processor.apply_chat_template(feature['conversations']) + self.processor.tokenizer.eos_token
#                 )
#             else:
#                 item_ids.append(str(feature['item_id']))
#                 texts.append(self.processor.apply_chat_template(feature['conversations'], add_generation_prompt=True,tokenize=True,))

#         # Build multi_modal_data
#         for vid in videos:
#             multi_modal_data.append({"video": [vid]})

#         # Return batch_dict compatible with vLLM
#         batch_dict = {
#             "prompts": texts,  # keep as string prompts
#             "multi_modal_data": multi_modal_data,
#             "item_ids": item_ids
#         }

#         return batch_dict

@dataclass
class NovaVllmCollator:
    data_type: str = 'bf16'
    processor: object = None
    tokenizer: object = None
    train_mode: bool = False
    input_start_id: int = 3
    input_end_id: int = 4

    def __post_init__(self):
        pass

    def __call__(self, features):
        item_ids, images, videos, texts = [], [], [], []

        for feature in features:
            images.extend(feature['images'])
            videos.extend(feature['videos'])

            if self.train_mode:
                texts.append(
                    self.processor.apply_chat_template(feature['conversations']) + self.tokenizer.eos_token)
            else:
                item_ids.append(str(feature['item_id']))
                texts.append(self.processor.apply_chat_template(feature['conversations'], add_generation_prompt=True))

        if not images:
            images = None
        if not videos:
            videos = None
        batch_dict = {}
        raw_prompt_ids = self.tokenizer.batch_encode_plus([i for i in texts])['input_ids']
        # print(self.tokenizer)
        # raw_prompt_ids = [self.processor.tokenizer.encode(text) for text in texts]
        # print(self.tokenizer.encode(texts[0]))
        #print(self.processor.tokenizer.batch_encode_plus([i for i in texts])['input_ids'])
        # model_inputs = self.processor(
        #         text=texts,
        #         images=images,
        #         videos=videos,
        #         padding=True,
        #         return_tensors="pt"
        #     )

        # raw_prompt_ids = model_inputs["input_ids"].tolist()

        # # print(texts)
        # print(len(videos), len(videos[0]))
        multi_modal_data = []
        for i in range(len(videos)):
            multi_modal_data.append({
                "video": [videos[i]]
            })
        batch_dict.update({'raw_prompt_ids': raw_prompt_ids})
        # batch_dict.update({'prompts': [i for i in texts]})
        batch_dict.update({'multi_modal_data': multi_modal_data})
        batch_dict.update({'item_ids': item_ids})
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

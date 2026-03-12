# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import copy
import logging
import os
import re
from collections import defaultdict
from typing import Optional
import bytedtos
from PIL import Image
import zipfile
import json

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)
import io
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
def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        train_mode: bool = False,
        max_frames: int = 32,
        num_for_eval: int = -1
    ):
        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.train_mode = train_mode
        self.max_frames = max_frames
        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 26000)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.num_for_eval = num_for_eval

        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        self.dataframe = json.load(open(self.data_files))
        if not self.train_mode:
            # 推理状态下，仅保留问题部分
            print(f"### infer model, will filter assistant part.....")
            for line in self.dataframe:
                line["conversations"] = line["conversations"][:1]
        if self.num_for_eval > 0:
            self.dataframe = self.dataframe[:self.num_for_eval]
        self.data_len = len(self.dataframe)
    
    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def get_from_tos(self, filepath):
        # the start_index corresponds to the local tos
        resp = read_zip_as_tos(filepath)
        # resp = self.tos_client_list[index % len(self.cluster_list)].get_object(filepath)
        raw_data = io.BytesIO(resp.data)
        images = []
        with zipfile.ZipFile(raw_data, 'r') as zip_file:
            for zip_image in zip_file.namelist():
                with zip_file.open(zip_image) as image_data:
                    image = Image.open(io.BytesIO(image_data.read()))
                    images.append(image.convert('RGB'))
        return images
    
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

    def _build_messages(self, idx):
        while True:
            # copy the item to avoid storing images and videos in memory
            item = copy.deepcopy(self.dataframe[idx])
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


    def _build_messages_dummy(self, idx):
        item = copy.deepcopy(self.dataframe[idx])
        videos = []
        for i in range(self.max_frames):
            random_array = np.random.randint(0, 256, (360, 640, 3), dtype=np.uint8)
            img = Image.fromarray(random_array, 'RGB')
            videos.append(img)
        videos = [videos]
        item.update({'images': [], 'videos': videos})
        return item
    

    def __getitem__(self, idx):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe[idx]
        messages = self._build_messages(idx)
        # messages = self._build_messages_dummy(idx)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            # raw_prompt = self.processor.apply_chat_template(
            #     messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            # )
            if self.train_mode:
                raw_prompt = self.processor.apply_chat_template(messages['conversations']) + self.tokenizer.eos_token
            else:
                raw_prompt = self.processor.apply_chat_template(messages['conversations'], add_generation_prompt=True)
            multi_modal_data = {}

            if len(messages['images']) == 0:
                images = None
            if len(messages['videos']) == 0:
                videos = None

            images = None
            row_dict_images = messages['images']
            if row_dict_images:
                images = copy.deepcopy(row_dict_images)
                # images = [process_image(image) for image in row_dict_images]
                for image in row_dict_images:
                    image.append(
                        process_image({
                            'image': image,
                            'min_pixels': self.processor.image_processor.min_pixels,
                            'max_pixel': self.processor.image_processor.max_pixels,
                        })
                    )
                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            # videos = None
            row_dict_videos = messages['videos']
            if row_dict_videos:
                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                try:
                    multi_modal_data["video"] = [np.array(video) for video in row_dict_videos]
                except: 
                    # sometimes each frame in the video do not have same resolution
                    std_size = row_dict_videos[0][0].size
                    for i, video in enumerate(row_dict_videos):
                        row_dict_videos[i] = [frame.resize(std_size, resample=Image.Resampling.BICUBIC) for frame in video]
                    multi_modal_data["video"] = [np.array(video) for video in row_dict_videos]
                # videos = copy.deepcopy(row_dict_videos)

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=row_dict_videos, padding=True, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            #addhoc, force to bf16
            for key in model_inputs.keys():
                if model_inputs[key].dtype == torch.float32:
                     model_inputs[key] = model_inputs[key].type(torch.bfloat16)
            
            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            # NotImplementedError()
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and ( \
            "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__ or \
            "NovaImageProcessor" in self.processor.image_processor.__class__.__name__):
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        row_dict['reward_model'] = {
                   "style": "rule",
                   "ground_truth": row_dict['reward_label']
               }
        row_dict['data_source'] = 'themis'
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()

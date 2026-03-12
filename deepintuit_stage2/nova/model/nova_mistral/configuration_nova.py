# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Nova model configuration"""

import warnings
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging, is_torch_available

logger = logging.get_logger(__name__)


class NovaVisionConfig(PretrainedConfig):
    model_type = "nova"
    base_config_key = "vision_config"

    def __init__(
            self,
            depth=27,
            embed_dim=1152,
            hidden_size=3584,
            hidden_act="quick_gelu",
            mlp_ratio=4,
            num_heads=16,
            in_channels=3,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
            intermediate_size=4304,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.intermediate_size = intermediate_size


class NovaConfig(PretrainedConfig):
    model_type = "nova"
    sub_configs = {"vision_config": NovaVisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            vocab_size=152064,
            hidden_size=8192,
            intermediate_size=29568,
            num_hidden_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,
            hidden_act="silu",
            max_position_embeddings=32768,
            initializer_range=0.02,
            rms_norm_eps=1e-05,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=1000000.0,
            use_sliding_window=False,
            sliding_window=4096,
            max_window_layers=80,
            attention_dropout=0.0,
            vision_config=None,
            rope_scaling=None,
            **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = NovaVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = NovaVisionConfig()

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        # and change type from 'mrope' to 'default' because `mrope` does defeault RoPE calculations
        # one can set it to "linear"/"dynamic" etc. to have scaled RoPE
        # TODO: @raushan update config in the hub
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self, ignore_keys={"mrope_section"})

        # Attributes with defaults
        self.tie_word_embeddings = tie_word_embeddings
        self.return_dict = kwargs.pop("return_dict", True)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.torchscript = kwargs.pop("torchscript", False)  # Only used by PyTorch models
        self.torch_dtype = kwargs.pop("torch_dtype", None)  # Only used by PyTorch models
        self.use_bfloat16 = kwargs.pop("use_bfloat16", False)
        self.tf_legacy_loss = kwargs.pop("tf_legacy_loss", False)  # Only used by TensorFlow models
        self.pruned_heads = kwargs.pop("pruned_heads", {})
        self.chunk_size_feed_forward = kwargs.pop("chunk_size_feed_forward", 0)

        # Is decoder is used in encoder-decoder models to differentiate encoder from decoder
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
        self.is_decoder = kwargs.pop("is_decoder", False)
        self.cross_attention_hidden_size = kwargs.pop("cross_attention_hidden_size", None)
        self.add_cross_attention = kwargs.pop("add_cross_attention", False)
        self.tie_encoder_decoder = kwargs.pop("tie_encoder_decoder", False)

        # Retrocompatibility: Parameters for sequence generation. While we will keep the ability to load these
        # parameters, saving them will be deprecated. In a distant future, we won't need to load them.
        for parameter_name, default_value in self._get_global_generation_defaults().items():
            setattr(self, parameter_name, kwargs.pop(parameter_name, default_value))

        # Fine-tuning task arguments
        self.architectures = kwargs.pop("architectures", None)
        self.finetuning_task = kwargs.pop("finetuning_task", None)
        self.id2label = kwargs.pop("id2label", None)
        self.label2id = kwargs.pop("label2id", None)
        if self.label2id is not None and not isinstance(self.label2id, dict):
            raise ValueError("Argument label2id should be a dictionary.")
        if self.id2label is not None:
            if not isinstance(self.id2label, dict):
                raise ValueError("Argument id2label should be a dictionary.")
            num_labels = kwargs.pop("num_labels", None)
            if num_labels is not None and len(self.id2label) != num_labels:
                logger.warning(
                    f"You passed along `num_labels={num_labels}` with an incompatible id to label map: "
                    f"{self.id2label}. The number of labels wil be overwritten to {self.num_labels}."
                )
            self.id2label = {int(key): value for key, value in self.id2label.items()}
            # Keys are always strings in JSON so convert ids to int here.
        else:
            self.num_labels = kwargs.pop("num_labels", 2)

        if self.torch_dtype is not None and isinstance(self.torch_dtype, str):
            # we will start using self.torch_dtype in v5, but to be consistent with
            # from_pretrained's torch_dtype arg convert it to an actual torch.dtype object
            if is_torch_available():
                import torch

                self.torch_dtype = getattr(torch, self.torch_dtype)

        # Tokenizer arguments TODO: eventually tokenizer and models should share the same config
        self.tokenizer_class = kwargs.pop("tokenizer_class", None)
        self.prefix = kwargs.pop("prefix", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.sep_token_id = kwargs.pop("sep_token_id", None)

        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        # task specific arguments
        self.task_specific_params = kwargs.pop("task_specific_params", None)

        # regression / multi-label classification
        self.problem_type = kwargs.pop("problem_type", None)
        # TPU arguments
        if kwargs.pop("xla_device", None) is not None:
            logger.warning(
                "The `xla_device` argument has been deprecated in v4.4.0 of Transformers. It is ignored and you can "
                "safely remove it from your `config.json` file."
            )

        # Name or path to the pretrained checkpoint
        self._name_or_path = str(kwargs.pop("name_or_path", ""))
        # Config hash
        self._commit_hash = kwargs.pop("_commit_hash", None)

        # Attention implementation to use, if relevant.
        self._attn_implementation_internal = kwargs.pop("attn_implementation", None)
        self._attn_implementation_autoset = False

        # Drop the transformers version info
        self.transformers_version = kwargs.pop("transformers_version", None)

        # Deal with gradient checkpointing
        if kwargs.get("gradient_checkpointing", False):
            warnings.warn(
                "Passing `gradient_checkpointing` to a config initialization is deprecated and will be removed in v5 "
                "Transformers. Using `model.gradient_checkpointing_enable()` instead, or if you are using the "
                "`Trainer` API, pass `gradient_checkpointing=True` in your `TrainingArguments`."
            )

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

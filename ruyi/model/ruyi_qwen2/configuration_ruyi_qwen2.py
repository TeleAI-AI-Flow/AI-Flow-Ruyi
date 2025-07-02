#!/usr/bin/env python
# Ref: https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/configuration_qwen2.py
# Copyright (c) Institute of Artificial Intelligence (TeleAI), China Telecom, 2025. All Rights Reserved.
"""RuyiQwen2 model configuration"""

import os 
import shutil 

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging


logger = logging.get_logger(__name__)


class RuyiQwen2Config(PretrainedConfig):

    model_type = "ruyi_qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `RuyiQwen2`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "eelayers.*.self_attn.q_proj": "colwise",
        "eelayers.*.self_attn_k_proj": "colwise",
        "eelayers.*.self_attn_v_proj": "colwise",
        "eelayers.*.self_attn_o_proj": "rowwise",
        "eelayers.*.mlp.gate_proj": "colwise",
        "eelayers.*.mlp.up_proj": "colwise",
        "eelayers.*.mlp.down_proj": "rowwise"
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "eelayers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        
        shared_heads=False,
        default_early_exit_point=-1, # [0, num_hidden_layers-1], -1 = num_hidden_layers - 1
        early_exit_points=list(range(1, 32, 2)),
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window  # we check `use_sliding_window` in the modeling code
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
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
        
        self.shared_heads = shared_heads
        self.default_early_exit_point = default_early_exit_point
        self.early_exit_points = early_exit_points
        self.auto_map = {
            "AutoConfig": "configuration_ruyi_qwen2.RuyiQwen2Config",
            "AutoModel": "modeling_ruyi_qwen2.RuyiQwen2Model",
            "AutoModelForCausalLM": "modeling_ruyi_qwen2.RuyiQwen2ForCausalLM"
        }

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
    
    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        shutil.copyfile(
            os.path.abspath(__file__), 
            os.path.join(save_directory, "configuration_ruyi_qwen2.py")
        )

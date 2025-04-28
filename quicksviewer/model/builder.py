#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from quicksviewer.model import *
from quicksviewer.model.utils import smart_tokenizer_and_embedding_resize
from quicksviewer import conversation as conversation_lib

def load_pretrained_model(model_path, model_base, model_name, version, load_8bit=False, load_4bit=False, overwrite_config=None, vpm_device=0, llm_device=0):
    device = torch.device(f'cuda:{vpm_device}')
    if vpm_device != llm_device: # Load onto CPU first
        device = vpm_device
    kwargs = {"device_map": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        # kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    else:
        # kwargs["torch_dtype"] = torch.float16
        kwargs["torch_dtype"] = torch.bfloat16

    if "quicksviewer" in model_name.lower():
        # Load Quicksviewer model
        if model_base is not None:
            # this may be mm projector only
            print("Loading Quicksviewer from base model...")
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            cfg_pretrained.vocab_size = len(tokenizer)
            cfg_pretrained.pad_token_id = None if cfg_pretrained.pad_token_id >= len(tokenizer) else cfg_pretrained.pad_token_id
            cfg_pretrained.pretrain_mm_adapter = None
            if overwrite_config is not None:
                print(f"Overwriting config with {overwrite_config}")
                for k, v in overwrite_config.items():
                    setattr(cfg_pretrained, k, v)
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            if "llama_3" in version.lower():
                tokenizer.pad_token = tokenizer.unk_token
                if tokenizer.pad_token is None:
                    smart_tokenizer_and_embedding_resize(
                        special_tokens_dict=dict(pad_token="[PAD]"),
                        tokenizer=tokenizer,
                        model=model,
                    )
                model.config.pad_token_id = tokenizer.pad_token_id
            model.initialize_vision_tokenizer(cfg_pretrained, tokenizer)
            mm_projector_weights = torch.load(os.path.join(model_path, "mm_projector.bin"), map_location="cpu")
            mm_projector_weights = {k: v.to(device) for k, v in mm_projector_weights.items()}
            info = model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if "llama_3" in version.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                if overwrite_config is not None:
                    print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(cfg_pretrained, k, v)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else: # default: qwen2
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                if overwrite_config is not None:
                    print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(cfg_pretrained, k, v)
                model = LlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
        # Setup common settings for VLM
        conversation_lib.default_conversation = conversation_lib.conv_templates[version] # Set conversation template

    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            use_fast = False
            if "mpt" in model_name.lower().replace("prompt", ""):
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 16384

    if vpm_device != llm_device: # inference on multi-gpus
        from accelerate import load_checkpoint_and_dispatch, init_empty_weights, infer_auto_device_map, dispatch_model
        device_map = infer_auto_device_map(model, max_memory={0: "80GB", 1: "80GB"},
            no_split_module_classes=['vision_model', 'LlamaDecoderLayer'])
        device_map["model.vision_tower"] = vpm_device # Vision
        device_map["model.vision_resampler"] = vpm_device
        device_map["model.cubing"] = vpm_device
        device_map["model.embed_tokens"] = vpm_device
        device_map["lm_head"] = vpm_device
        device_map["model.layers"] = llm_device # LLM
        for _ in range(len(model.model.layers)):
            device_map[f"model.layers.{_}"] = llm_device
            device_map[f"model.layers.{_}.self_attn"] = llm_device
            device_map[f"model.layers.{_}.self_attn.q_proj"] = llm_device
            device_map[f"model.layers.{_}.self_attn.k_proj"] = llm_device
            device_map[f"model.layers.{_}.self_attn.v_proj"] = llm_device
            device_map[f"model.layers.{_}.self_attn.o_proj"] = llm_device
            device_map[f"model.layers.{_}.self_attn.rotary_emb"] = llm_device
            device_map[f"model.layers.{_}.mlp"] = llm_device
            device_map[f"model.layers.{_}.mlp.gate_proj"] = llm_device
            device_map[f"model.layers.{_}.mlp.up_proj"] = llm_device
            device_map[f"model.layers.{_}.mlp.down_proj"] = llm_device
            device_map[f"model.layers.{_}.input_layernorm"] = llm_device
            device_map[f"model.layers.{_}.post_attention_layernorm"] = llm_device
            skip_keys = ['inputs_embeds', 'position_ids', 'cache_position', 'attention_mask']
        device_map["model.norm"] = llm_device
        device_map["model.rotary_emb"] = llm_device

        model = dispatch_model(model, device_map=device_map, skip_keys=skip_keys)
        model.eval()

    return tokenizer, model, image_processor, context_len

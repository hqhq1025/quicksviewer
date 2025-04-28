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


import inspect
from typing import List, Optional, Tuple, Union
from copy import deepcopy
from PIL import Image
import numpy as np
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from quicksviewer.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from quicksviewer.train.utils import calculate_loss_weight
from quicksviewer.constants import DEFAULT_IMAGE_TOKEN
from quicksviewer.data.preprocess import preprocess_multimodal_image, preprocess_multimodal_video, preprocess
from quicksviewer.utils.mm_utils import process_images
from quicksviewer.utils.utils import batch_to
from quicksviewer.train.sequence_parallel import set_pg_manager
from quicksviewer.model.utils import patch, set_seqlens_in_batch




class LlavaConfig(LlamaConfig):
    model_type = "quicksviewer_llama"
    


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        LlamaForCausalLM.__init__(self, config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Qiji: add for SP
        patch(self.model)

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        prompts: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None,
        image_sizes: Optional[List[List[int]]] = None,
        imidx_in_multi: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[bool] = None,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        videos_bound: Optional[List[List[int]]] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            lr_gumbel = kwargs.get('lr_gumbel', 0.1)
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, seqlens_in_batch, list_z) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes, prompts, imidx_in_multi,
                seqlens_in_batch, videos_bound, lr_gumbel
            )

        if seqlens_in_batch is None:
                seqlens_in_batch = torch.sum(attention_mask, dim=1)
        set_seqlens_in_batch(seqlens_in_batch)

        need_repack = kwargs.get('need_repack', False)
        if self.training and need_repack and inputs_embeds is not None:
            (
                _,
                new_position_ids,
                new_attention_mask,
                _,
                new_inputs_embeds,
                new_labels,
                sorted_seqlens_in_batch,
            ) = self.repack_multimodal_data(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            )
            if sorted_seqlens_in_batch is None:
                sorted_seqlens_in_batch = seqlens_in_batch
            if sorted_seqlens_in_batch is not None:
                set_seqlens_in_batch(sorted_seqlens_in_batch)
            new_input_ids = None
            past_key_values = None
        else:
            new_attention_mask = attention_mask
            new_position_ids = position_ids
            new_inputs_embeds = inputs_embeds
            new_labels = labels
            sorted_seqlens_in_batch = attention_mask.sum(-1).int() if need_repack else seqlens_in_batch
            new_input_ids = input_ids

        outputs = super().forward(
            input_ids=new_input_ids,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=new_inputs_embeds,
            labels=new_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Loss rescale for SP & DP loss match
        if new_labels is not None:
            loss_z_tot = [torch.norm(z,2,dim=-1) for z in list_z if z is not None] # version-2: 2-norm(z)
            loss_z = torch.cat(loss_z_tot).mean() if len(loss_z_tot)>0 else 0.0
            # print(f"[loss_z]: {loss_z}")

            loss_weight = calculate_loss_weight(new_labels)
            outputs.loss = outputs.loss * loss_weight + loss_z * 0.001
        return outputs


    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        prompts: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None,
        image_sizes: Optional[List[List[int]]] = None,
        imidx_in_multi: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[bool] = None,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        videos_bound: Optional[List[List[int]]] = None,
        tokenizer=None,
        terminators=['<|eot_id|>'],
        llm_device: Optional[torch.device] = None,
        lr_gumbel: Optional[int] = 0.0, # Qiji: default no Gumbel noise
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            labels = input_ids.clone()
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, seqlens_in_batch, list_z) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes, prompts, imidx_in_multi,
                seqlens_in_batch, videos_bound, lr_gumbel
            )

        # For inference model on different devices
        input_ids = None
        if llm_device is not None:
            inputs_embeds = inputs_embeds.to(llm_device)
            attention_mask = attention_mask.to(llm_device) if attention_mask is not None else None
            input_ids = torch.ones((inputs_embeds.shape[0], 0), dtype=torch.long, device=llm_device)

        output_ids = super().generate(
            input_ids = input_ids,
            # input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            seqlens_in_batch= seqlens_in_batch,
            **kwargs,
        )

        # print(f"----- GENERATE\n output_ids: {output_ids.tolist()}")
        from quicksviewer.train.sequence_parallel import get_pg_manager
        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            sp_degree = PROCESS_GROUP_MANAGER.sp_degree
            sp_rank = PROCESS_GROUP_MANAGER.sp_rank
            sp_group = PROCESS_GROUP_MANAGER.sp_pg
            ring_degree = PROCESS_GROUP_MANAGER.ring_degree
            ring_rank = PROCESS_GROUP_MANAGER.ring_rank
            ring_type = PROCESS_GROUP_MANAGER.ring_type
            ulysses_degree = PROCESS_GROUP_MANAGER.ulysses_degree
            ulysses_rank = PROCESS_GROUP_MANAGER.ulysses_rank

            bs, shard_seqlen = output_ids.shape
            sp_seq_len = [torch.zeros(1, dtype=torch.int64, device=output_ids.device) for _ in range(sp_degree)]
            dist.all_gather(sp_seq_len, torch.tensor(shard_seqlen, device=output_ids.device), group=sp_group)
            sp_output_ids = [torch.zeros(bs, slen, dtype=torch.int64, device=output_ids.device) for slen in sp_seq_len]
            dist.all_gather(sp_output_ids, output_ids, group=sp_group)
            res_output_ids = []
            for b, shard_ids in enumerate(zip(*sp_output_ids)):
                res_output_ids.append(torch.cat([out_ids[out_ids!=tokenizer.pad_id] for out_ids in shard_ids], dim=0))
            output_ids = res_output_ids
            print(f"----- GENERATE\n sp_output_ids: {[out_ids.tolist() for out_ids in sp_output_ids]}\n output_ids: {output_ids.tolist()}\n sp_output_ids_decode: {[tokenizer.convert_ids_to_tokens(out_ids.tolist()) for out_ids in sp_output_ids]}")

        # decode text
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in terminators]
        result_text = []
        for result in output_ids:
            result = result[result != 0]
            if result[0] == tokenizer.bos_token_id:
                result = result[1:]
            if result[-1] in terminators:
                result = result[:-1]
            result_text.append(tokenizer.decode(result).strip())
        return result_text


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs
    
    
    @torch.no_grad()
    def chat(self,
            image,
            msgs, # [[{'from':'human'|'gpt','value':str|tuple, 'timestamps':list}]]
            modalities, # ['image', 'video', ..]
            tokenizer,
            image_processor,
            max_new_tokens=2048,
            # seq_parallel_size=1,
            dtype=None,
            **kwargs,
    ) -> Tuple[str, list, list]:
        """ Given user messages composed of interleaved image-prompt,
              respond the model output, past_key_values, and vision_hidden_states.
            Note that we assume each conversation refer to only one modality, either n images or a video!
          Params:
            @msgs: [[{'from': 'human', 'value': (Image, Image, 'Hello.')}, {'from': 'gpt', 'value': 'Hi!'}], ..]
            @image: [[Image, ], ..] or None
        """
        generation_config = {
            "tokenizer": tokenizer,
            "max_new_tokens": max_new_tokens,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id
        }
        
        if isinstance(msgs[0], list):
            batched = True
        else:
            batched = False
        msgs_list = msgs
        images_list = image
        
        if batched is False:
            images_list, msgs_list, modalities = [images_list], [msgs_list], [modalities]
        assert len(images_list) == len(msgs_list), "The batch dim of images_list and msgs_list should be the same."

        # batch = []
        bsz_inputs, bsz_images, bsz_image_sizes, bsz_imidx_in_multi = [], [], [], []
        for i, (image, msgs, modality) in enumerate(zip(images_list, msgs_list, modalities)):
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            copy_msgs = deepcopy(msgs)

            if image is not None and isinstance(copy_msgs[0]["value"], str):
                copy_msgs[0]["value"] = [image, copy_msgs[0]["value"]] # Add image to form a tuple value

            cut = 2
            copy_msgs_rou = [copy_msgs[_:_+cut] for _ in range(0, len(copy_msgs), cut)]
            timestamps = None
            msgs_images = []
            for ii, rou_msgs in enumerate(copy_msgs_rou):
                rou_images = []
                for iii, msg in enumerate(rou_msgs):
                    role = msg["from"]
                    value = msg["value"]
                    timestamps = msg.get('timestamps', None) if 'timestamps' in msg else timestamps
                    assert role in ["human", "gpt"]
                    if iii == 0:
                        assert role == "human", "The role of first msg should be human"
                    if isinstance(value, str):
                        value = [value]
                    cur_msgs = []
                    for c in value:
                        if isinstance(c, Image.Image) and modality=='image':
                            cur_msgs.append(DEFAULT_IMAGE_TOKEN)
                            rou_images.append(c)
                        elif isinstance(c, np.ndarray) and c.ndim==4 and modality=='video':
                            cur_msgs.append(DEFAULT_IMAGE_TOKEN)
                            rou_images.extend([Image.fromarray(f) for f in c])
                        elif isinstance(c, str):
                            c = c.replace(DEFAULT_IMAGE_TOKEN, '')
                            cur_msgs.append(c)
                    msg["value"] = "\n".join(cur_msgs)
                msgs_images.extend(rou_images)

            # Process a conversations with multi-rounds
            images, conversations, image_sizes, imidx_in_multi = None, None, None, []
            image_sizes = [img.size for img in msgs_images]
            setattr(self.model.config, 'is_multimodal', True)
            if modality == 'image':
                images, pathnums_imgs = process_images(msgs_images, image_processor, self.model.config, return_pathnums=True)
                imidx_in_multi = np.cumsum([0] + [images[_].shape[0] for _ in range(len(images)-1)]).tolist()
                conversations = preprocess_multimodal_image([copy_msgs], self.model.config, pathnums_imgs)
            elif modality == 'video':
                images = process_images(msgs_images, image_processor, self.model.config, image_aspect_ratio='original', return_pathnums=False)
                imidx_in_multi = list(range(len(imidx_in_multi), len(imidx_in_multi)+len(images))) # Be careful
                conversations = preprocess_multimodal_video([copy_msgs], self.model.config, frame_timestamps=timestamps, nframes=len(images),is_cubing=True, add_thumbnail=self.model.config.mm_use_thumbnail)
            images = images.view(-1, *images.shape[-3:])
            if dtype is not None:
                images = images.to(dtype=dtype)
            
            input_ids = preprocess(
                conversations,
                tokenizer,
                has_image=len(images) > 0,
                prompt=None,
                build_labels=False)['input_ids'][0]
            # labels = input_ids.clone()
            
            # Collect for a batch
            bsz_inputs.append(input_ids)
            bsz_images.append(images)
            bsz_image_sizes.append(image_sizes)
            bsz_imidx_in_multi.append(imidx_in_multi)

        # Collator for a batch
        batch = {}
        batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(
            bsz_inputs,
            batch_first=True,
            padding_value=tokenizer.pad_token_id)[:, :tokenizer.model_max_length].to(torch.device('cuda:0'))
        if len(bsz_images)>0:
            if all(x is not None and x.shape == bsz_images[0].shape for x in bsz_images) and len(bsz_images) > 1:
                batch['images'] = torch.stack(bsz_images).to(torch.device('cuda:0'))
            else:
                batch['images'] = [img.to(torch.device('cuda:0')) for img in bsz_images]
        batch['modalities'] = modalities
        batch['attention_mask'] = batch['input_ids'].ne(tokenizer.pad_token_id)
        batch['image_sizes'] = bsz_image_sizes
        batch['imidx_in_multi'] = bsz_imidx_in_multi

        return self.generate(**batch, **generation_config, **kwargs)

        # if seq_parallel_size > 1:
        #     from llavacube.train.train_hybrid import set_sp_environment
        #     set_pg_manager(seq_parallel_size, -1, ring_type=None)
        #     collator = DataCollatorForSupervisedDatasetSeqParallel(tokenizer)
        #     batch = collator(batch)
        #     with set_sp_environment():
        #         return self.generate(**batch, **kwargs)
        # else:
        #     collator = DataCollatorForSupervisedDataset(tokenizer)
        #     batch = collator(batch)
        #     return self.generate(**batch, **kwargs)




if LlavaConfig.model_type == "quicksviewer":
    LlavaConfig.model_type = "quicksviewer_llama"

AutoConfig.register("quicksviewer_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)

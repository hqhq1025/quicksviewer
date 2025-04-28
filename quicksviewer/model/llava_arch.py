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

import warnings
from abc import ABC, abstractmethod

import ipdb.stdout
import torch.distributed as dist

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from quicksviewer.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from quicksviewer.constants import DEFAULT_PATCH_START_TOKEN, DEFAULT_PATCH_END_TOKEN, DEFAULT_VIDEO_START_TOKEN, DEFAULT_VIDEO_END_TOKEN

from quicksviewer.utils.mm_utils import get_anyres_image_grid_shape
from quicksviewer.train.sequence_parallel import get_pg_manager
from quicksviewer.model.multimodal_cubing.builder import build_cubing



def update_placeholders_by_cube(cube_bound, video_bound, input_ids, attention_mask, labels, position_ids=None, seqlens_in_batch=None, IMID=-200, PADID=128256, NUM_TOKENS_PER_IMAGE=64):
    """ 1. input_ids/labels: merge multiple <image> in one cube, 2. position_ids: merge and shift, 3. attention_mask: merge and shift.
      Assume video bounds, e.g., [[0, 4), [4, 8), ...]
      Assume the cube bounds are consecutive not overlaps, e.g., [[[0, 3), [3, 4)],  [[0, 4)], ...]
    """
    if not video_bound or not cube_bound:
        return input_ids, labels, attention_mask, position_ids, seqlens_in_batch

    bs, seq_len = input_ids.shape
    new_input_ids, new_labels, new_attention_mask, new_position_ids = [], [], [], []
    b, g = 0, 0
    while b < input_ids.size(0):
        cur_input_ids, cur_labels, cur_attention_mask, cur_position_ids = [], [], [], []
        i, p, pos_off = 0, 0, 0
        while i < input_ids.size(1) and input_ids[b,i]!=PADID: # WARNING: Set terminating line as PADID!
            if input_ids[b, i] == IMID:
                cbounds = None
                for ii, vbound in enumerate(video_bound):
                    if vbound[0] <= g < vbound[1]: # is a video frame
                        break
                if vbound[0] <= g < vbound[1]:
                    cbounds = cube_bound[ii] # found N bounds of cubes
                # Merge
                if cbounds is not None:
                    assert g == cbounds[0][0]+ vbound[0], f"The image number in input_ids and in cube_bound do not match!\n input_ids:{input_ids.tolist()}\n video_bound: {video_bound}\n cube_bound: {cube_bound}\n g: {g}"
                    for cb in cbounds:
                        assert cb[0]+vbound[0] <= g < cb[1] + vbound[0]
                        im_added = False
                        # Merge
                        while g < cb[1] + vbound[0] and i < input_ids.size(1) and input_ids[b,i]!=PADID: # WARNING: Set terminating line as PADID!
                            if (input_ids[b, i] == IMID and not im_added) or (input_ids[b, i] != IMID and g<= cb[0] + vbound[0]): # tokens before left boundary
                                cur_input_ids.append(input_ids[b,i])
                                cur_labels.append(labels[b,i])
                                cur_attention_mask.append(attention_mask[b,i])
                                if input_ids[b,i] == IMID:
                                    im_added = True
                                    g += 1 # accumulate frame
                                    if position_ids is not None:
                                        pos_off = 0 if position_ids[b, p] <= 0 else pos_off
                                        cur_position_ids.extend([ _- pos_off for _ in position_ids[b, p: p+NUM_TOKENS_PER_IMAGE]]) # update positions of image tokens
                                        if any([ _- pos_off<-1 for _ in position_ids[b, p: p+NUM_TOKENS_PER_IMAGE]]):
                                            assert 1==2
                                        p += NUM_TOKENS_PER_IMAGE
                                else:
                                    if position_ids is not None:
                                        pos_off = 0 if position_ids[b, p] <= 0 else pos_off
                                        cur_position_ids.append(position_ids[b, p] - pos_off)
                                        if position_ids[b, p] - pos_off<-1:
                                            assert 1==2
                                        p += 1
                            elif i != input_ids.size(1)-1: # Reduce here
                            # Reduce when i is not at the last position, elsewise will end the 2nd while-loop based on i
                                if input_ids[b, i] == IMID:
                                    g += 1 # accumulate frame
                                    if position_ids is not None:
                                        pos_off = pos_off+sum([position_ids[b,p+_]-position_ids[b,p+_-1] for _ in range(1,NUM_TOKENS_PER_IMAGE)])+1 # update with deltas
                                        p += NUM_TOKENS_PER_IMAGE
                                else:
                                    if position_ids is not None:
                                        pos_off = pos_off+(position_ids[b,p+1]-position_ids[b,p])
                                        p += 1
                            i +=1
                else:
                    cur_input_ids.append(input_ids[b,i])
                    cur_labels.append(labels[b,i])
                    cur_attention_mask.append(attention_mask[b,i])
                    i += 1
                    g += 1 # accumulate image
                    if position_ids is not None:
                        pos_off = 0 if position_ids[b, p] <= 0 else pos_off
                        cur_position_ids.extend([ _- pos_off for _ in position_ids[b, p: p+NUM_TOKENS_PER_IMAGE]]) # update positions of image tokens
                        if any([ _- pos_off<-1 for _ in position_ids[b, p: p+NUM_TOKENS_PER_IMAGE]]):
                            assert 1==2
                        p += NUM_TOKENS_PER_IMAGE
            else:
                cur_input_ids.append(input_ids[b,i])
                cur_labels.append(labels[b,i])
                cur_attention_mask.append(attention_mask[b,i])
                i += 1
                if position_ids is not None:
                    pos_off = 0 if position_ids[b, p] <= 0 else pos_off
                    cur_position_ids.append(position_ids[b, p] - pos_off)
                    if position_ids[b, p] - pos_off <-1:
                        assert 1==2
                    p += 1
        b+=1
        new_input_ids.append(torch.stack(cur_input_ids))
        new_labels.append(torch.stack(cur_labels))
        new_attention_mask.append(torch.stack(cur_attention_mask))
        if position_ids is not None:
            _i = len(cur_position_ids)-1
            while _i>=0 and cur_position_ids[_i]==-1: _i-=1
            cur_position_ids = cur_position_ids[:_i+1] # remove excess paddings at the end
            new_position_ids.append(torch.stack(cur_position_ids))
    new_input_ids = torch.nn.utils.rnn.pad_sequence(new_input_ids, batch_first=True, padding_value=PADID)
    new_labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_INDEX)
    new_attention_mask = torch.nn.utils.rnn.pad_sequence(new_attention_mask, batch_first=True, padding_value=False)
    new_position_ids = torch.nn.utils.rnn.pad_sequence(new_position_ids, batch_first=True, padding_value=-1) if position_ids is not None else position_ids
    new_seqlens_in_batch = seqlens_in_batch
    if seqlens_in_batch is not None:
        assert new_position_ids is not None
        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            sp_group = PROCESS_GROUP_MANAGER.sp_pg
            sp_degree = PROCESS_GROUP_MANAGER.sp_degree
            sp_rank = PROCESS_GROUP_MANAGER.sp_rank
            # get
            bs, shard_seqlen = new_position_ids.shape
            sp_seq_len = [torch.zeros(1, dtype=torch.int64, device=new_position_ids.device) for _ in range(sp_degree)]
            dist.all_gather(sp_seq_len, torch.tensor(shard_seqlen, device=new_position_ids.device), group=sp_group)
            sp_new_position_ids = [torch.full((bs, sp_seq_len[_]), -1, dtype=torch.int64, device=new_position_ids.device) for _ in range(sp_degree)]
            dist.all_gather(sp_new_position_ids, new_position_ids, group=sp_group)
            sp_new_position_ids = torch.cat(sp_new_position_ids, dim=1)
        else:
            sp_new_position_ids = new_position_ids
        # update
        new_seqlens_in_batch = []
        for b, posids in enumerate(sp_new_position_ids):
            prev_pid = posids[0]
            for pid in posids:
                if pid == -1: continue
                if pid > prev_pid:
                    new_seqlens_in_batch[-1] +=1
                else: # new sample in batch
                    new_seqlens_in_batch.append(1)
        new_seqlens_in_batch = torch.tensor(new_seqlens_in_batch, dtype=seqlens_in_batch.dtype, device=seqlens_in_batch.device)
    return new_input_ids, new_labels, new_attention_mask, new_position_ids, new_seqlens_in_batch


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=False)
            self.vision_resampler = build_vision_resampler(config, mm_resampler_embeddim=self.config.hidden_size, mm_resampler_visiondim=self.vision_tower.hidden_size)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)
            
            self.cubing_type = getattr(config, 'mm_cubing', 'identity')
            self.cubing = build_cubing(
                config,
                self.vision_tower.hidden_size,
                self.vision_tower.num_patches,
                lm_dim = self.config.hidden_size
            )

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def initialize_vision_modules(self, model_args, fsdp=None, rebuild_vision=False):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_adapter = model_args.pretrain_mm_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        if self.get_vision_tower() is None or rebuild_vision:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, mm_resampler_embeddim=self.config.hidden_size, mm_resampler_visiondim=vision_tower.hidden_size)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
            vision_tower.load_model()

            self.cubing_type = getattr(model_args, 'mm_cubing', 'identity')
            self.cubing = build_cubing(
                model_args,
                self.vision_tower.hidden_size,
                self.vision_tower.num_patches,
                lm_dim = self.config.hidden_size
            )
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        if hasattr(model_args, 'cubing_vit_forward_n_layers'):
            self.cubing.forward_n_layers = model_args.cubing_vit_forward_n_layers
        print(f"#### Using forward_n_layers={self.cubing.forward_n_layers} for cubing.")
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        self.config.patchify_video_feature = getattr(model_args, "patchify_video_feature", False)

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            print(incompatible_keys)
            self.cubing.load_state_dict(get_w(mm_projector_weights, "cubing"))
        



def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_vision_resampler(self):
        return self.get_model().vision_resampler


    def encode_images(self, images, videos_bound, tgt_size, is_cubing=True, temperature=0.5, FPQ=5, lr_gumbel=0.1):
        cube_bound, list_z = None, []
        if is_cubing:
            image_features, videos_bound, cube_bound, list_z = self.get_model().cubing(self.get_model().get_vision_tower(), self.get_model().vision_resampler, images, tgt_size, videos_bound, temperature, FPQ, lr_gumbel)
        
        else:
            image_features = self.get_model().get_vision_tower()(images)
            image_features = self.get_model().vision_resampler(image_features, tgt_size=tgt_size)
            image_features = self.get_model().mm_projector(image_features)
        return image_features, videos_bound, cube_bound, list_z

    def update_prompt(self, prompts=None):
        self.prompts = prompts

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, modalities, image_sizes=None,prompts=None, imidx_in_multi=None,
        seqlens_in_batch=None, videos_bound=None, lr_gumbel=0.1,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, seqlens_in_batch, [None]

        if isinstance(modalities, str):
            modalities = [modalities]

        image_idx_in_batch, video_idx_in_batch = [], []
        for _ in range(len(modalities)):
            if modalities[_] == "image":
                image_idx_in_batch.append(_)
            elif modalities[_] == 'video':
                video_idx_in_batch.append(_)
        if type(images) is list or images.ndim == 5:

            images_list, _videos_bound, img_counts_batch, vid_counts_batch = [], [], [0]*len(images), [0]*len(images)
            prev = 0
            for i, image in enumerate(images):
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))
                if modalities[i] == 'video':
                    _videos_bound.append([prev, prev+image.shape[0]])
                    vid_counts_batch[i] += 1
                else:
                    img_counts_batch[i] += image.size(0)
                prev += image.shape[0]

            if videos_bound is None:
                videos_bound = _videos_bound

            concat_images = torch.cat(images_list, dim=0)

            # is_cubing =  self.get_model().cubing_type != 'identity'
            is_cubing =  self.get_model().config.is_cubing
            tgt_size = (self.get_vision_tower().num_patches_per_side, )*2
            image_features, videos_bound, cube_bound, list_z = self.encode_images(concat_images, videos_bound, tgt_size, is_cubing, temperature=0.5, lr_gumbel=lr_gumbel)
            input_ids_debug, attention_mask_debug, labels_debug, position_ids_debug, seqlens_in_batch_debug = input_ids, attention_mask, labels, position_ids, seqlens_in_batch
            if is_cubing:
                input_ids, labels, attention_mask, position_ids, seqlens_in_batch  = \
                    update_placeholders_by_cube(cube_bound, videos_bound, input_ids, attention_mask, labels, position_ids, seqlens_in_batch,PADID=self.config.pad_token_id)
                split_sizes = 1
            else:   
                split_sizes = [image.shape[0] for image in images_list]

            image_features = torch.split(image_features, split_sizes, dim=0)

            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            if mm_patch_merge_type == "flat":
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    new_image_features.extend(torch.split(image_feature, 1, dim=0))
            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # For video
                    if image_idx not in image_idx_in_batch:
                        new_image_features.extend(torch.split(image_feature, 1, dim=0))
                        continue

                    if image_feature.shape[0] > 1:
                        image_feature_multi = []
                        n_multi = len(imidx_in_multi[image_idx])
                        for ii, imid in enumerate(imidx_in_multi[image_idx]):
                            base_image_feature = image_feature[imid]
                            image_feature = image_feature[imid+1: imidx_in_multi[image_idx][ii+1] if ii<n_multi-1 else 1000]
                            height = width = int(self.get_vision_resampler().num_queries ** (1/2))
                            if image_aspect_ratio == "anyres":
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx][imid], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            else:
                                image_feature = image_feature.view(2, 2, height, width, -1)

                            if "maxpool2x2" in mm_patch_merge_type:
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = nn.functional.max_pool2d(image_feature, 2)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            elif "unpad" in mm_patch_merge_type:
                                #
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = unpad_image(image_feature, image_sizes[image_idx][imid])
                                image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            else:
                                image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                                image_feature = image_feature.flatten(0, 3)
                            if "nobase" in mm_patch_merge_type:
                                pass
                            else:
                                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                            image_feature_multi.append(image_feature)
                        image_feature = torch.cat(image_feature_multi, dim=0)
                    else:
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)
                    new_image_features.append(image_feature)
                # image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
            image_features = new_image_features
        else:
            # image_features = self.encode_images(images)
            image_features, cube_bound = self.encode_images(images)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_labels = labels[batch_idx]

            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                if cur_image_features.ndim == 3:
                    cur_image_features = cur_image_features.squeeze(0)
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                try:
                    cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                except:
                    import pdb
                    pdb.set_trace()
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    if cur_image_features.ndim == 3:
                        hidden_size = cur_image_features.shape[-1]
                        cur_image_features = cur_image_features.reshape(-1, hidden_size)
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            if any(len(x) > tokenizer_model_max_length for x in new_input_embeds):
                warnings.warn("Inputs truncated!")
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        
        gpu_rank = new_input_embeds.device.index if new_input_embeds.is_cuda else None
    
        # We will not use packing here when sequence parallelism is enabled.
        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            return (
                None,
                _position_ids,
                attention_mask,
                past_key_values,
                new_input_embeds,
                new_labels,
                seqlens_in_batch,
                list_z,
                # debug_feats
            )

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
            seqlens_in_batch,
            list_z
        )


    def repack_multimodal_data(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        inputs_embeds,
        labels,
    ):
        # Handle sequence parallelism
        PROCESS_GROUP_MANAGER = get_pg_manager()

        # We do re-sharding instead of packing here to ensure the sequence length is the same across all ranks.
        if PROCESS_GROUP_MANAGER is not None:
            sp_degree = PROCESS_GROUP_MANAGER.sp_degree
            sp_rank = PROCESS_GROUP_MANAGER.sp_rank
            sp_group = PROCESS_GROUP_MANAGER.sp_pg
            ring_degree = PROCESS_GROUP_MANAGER.ring_degree
            ring_rank = PROCESS_GROUP_MANAGER.ring_rank
            ring_type = PROCESS_GROUP_MANAGER.ring_type
            ulysses_degree = PROCESS_GROUP_MANAGER.ulysses_degree
            ulysses_rank = PROCESS_GROUP_MANAGER.ulysses_rank

            bs, shard_seqlen = position_ids.shape
            sp_seq_len = [torch.zeros(1, dtype=torch.int64, device=position_ids.device) for _ in range(sp_degree)]
            dist.all_gather(sp_seq_len, torch.tensor(shard_seqlen, device=position_ids.device), group=sp_group)
            sp_seq_len_cat = torch.cat(sp_seq_len, dim=0)

            if sp_rank == 0:
                original_start_id = 0
            else:
                original_start_id = torch.sum(sp_seq_len_cat[:sp_rank]).item()
            original_end_id = torch.sum(sp_seq_len_cat[: sp_rank + 1]).item()

            # Gather attention_mask, position_ids, labels and input_embeds
            all_inputs_embeds = torch.zeros(
                bs,
                torch.sum(sp_seq_len_cat),
                inputs_embeds.shape[-1],
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            ).contiguous()
            all_inputs_embeds[:, original_start_id:original_end_id, :] += inputs_embeds
            dist.barrier(group=sp_group)
            dist.all_reduce(all_inputs_embeds, group=sp_group)
            dist.barrier(group=sp_group)

            attention_mask_list = [
                torch.zeros((bs, sp_seq_len[i]), dtype=attention_mask.dtype, device=attention_mask.device)
                for i in range(sp_degree)
            ]
            position_ids_list = [
                torch.zeros((bs, sp_seq_len[i]), dtype=position_ids.dtype, device=position_ids.device)
                for i in range(sp_degree)
            ]
            labels_list = [
                torch.zeros((bs, sp_seq_len[i]), dtype=labels.dtype, device=labels.device) for i in range(sp_degree)
            ]

            dist.all_gather(attention_mask_list, attention_mask, group=sp_group)
            dist.all_gather(position_ids_list, position_ids, group=sp_group)
            dist.all_gather(labels_list, labels, group=sp_group)

            effective_seqlen_list = [attention_mask_list[i].sum(dim=-1) for i in range(sp_degree)]
            effective_seqlen = torch.stack(effective_seqlen_list, dim=-1)
            effective_seqlen_batch_list = torch.unbind(effective_seqlen, dim=0)

            global_attention_mask_list = []
            global_position_ids_list = []
            global_labels_list = []
            global_inputs_embeds_list = []
            for i in range(bs):
                global_attention_mask_batch_list = []
                global_position_ids_batch_list = []
                global_labels_batch_list = []
                global_inputs_embeds_batch_list = []
                for j in range(sp_degree):
                    eff_len = effective_seqlen_batch_list[i][j]
                    prev_len = torch.sum(sp_seq_len_cat[:j]).item() if j > 0 else 0

                    global_attention_mask_batch_list.append(attention_mask_list[j][i, :eff_len])
                    global_position_ids_batch_list.append(position_ids_list[j][i, :eff_len])
                    global_labels_batch_list.append(labels_list[j][i, :eff_len])
                    global_inputs_embeds_batch_list.append(all_inputs_embeds[i, prev_len : prev_len + eff_len, :])
                global_attention_mask_list.append(torch.cat(global_attention_mask_batch_list, dim=0))
                global_position_ids_list.append(torch.cat(global_position_ids_batch_list, dim=0))
                global_labels_list.append(torch.cat(global_labels_batch_list, dim=0))
                global_inputs_embeds_list.append(torch.cat(global_inputs_embeds_batch_list, dim=0))

                global_attention_mask = torch.nn.utils.rnn.pad_sequence(
                    global_attention_mask_list, batch_first=True, padding_value=False
                )
                global_position_ids = torch.nn.utils.rnn.pad_sequence(
                    global_position_ids_list, batch_first=True, padding_value=-1
                )
                global_labels = torch.nn.utils.rnn.pad_sequence(
                    global_labels_list, batch_first=True, padding_value=IGNORE_INDEX
                )
                global_inputs_embeds = torch.nn.utils.rnn.pad_sequence(
                    global_inputs_embeds_list, batch_first=True, padding_value=0
                )

            # Re-shard the inputs
            if ring_degree > 1:
                total_effective_seqlen = torch.sum(effective_seqlen, dim=1)
                new_seqlen_per_rank = total_effective_seqlen // sp_degree
                assert torch.all(
                    total_effective_seqlen % sp_degree == 0
                ), f"total_effective_seqlen of {total_effective_seqlen} must be divisible by sp_degree"

                max_new_seqlen = torch.max(new_seqlen_per_rank).item()

                new_attention_mask = torch.zeros(
                    (bs, max_new_seqlen), dtype=global_attention_mask.dtype, device=global_attention_mask.device
                )
                new_position_ids = torch.zeros(
                    (bs, max_new_seqlen), dtype=global_position_ids.dtype, device=global_position_ids.device
                )
                new_labels = torch.full(
                    (bs, max_new_seqlen), IGNORE_INDEX, dtype=global_labels.dtype, device=global_labels.device
                )
                new_inputs_embeds = torch.zeros(
                    (bs, max_new_seqlen, global_inputs_embeds.shape[-1]),
                    dtype=global_inputs_embeds.dtype,
                    device=global_inputs_embeds.device,
                )

                if ring_type == "ring_varlen":
                    for i in range(bs):
                        start_idx = new_seqlen_per_rank[i] * sp_rank
                        end_idx = start_idx + new_seqlen_per_rank[i]
                        new_attention_mask[i, : new_seqlen_per_rank[i]] = global_attention_mask[i, start_idx:end_idx]
                        new_position_ids[i, : new_seqlen_per_rank[i]] = global_position_ids[i, start_idx:end_idx]
                        new_labels[i, : new_seqlen_per_rank[i]] = global_labels[i, start_idx:end_idx]
                        new_inputs_embeds[i, : new_seqlen_per_rank[i], :] = global_inputs_embeds[
                            i, start_idx:end_idx, :
                        ]
                elif ring_type == "zigzag_ring_varlen":
                    chunk_size = total_effective_seqlen // (2 * sp_degree)
                    for i in range(bs):
                        # Zigzag pattern indices
                        if sp_degree == ring_degree:
                            forward_rank_idx = sp_rank
                            backward_rank_idx = 2 * sp_degree - sp_rank - 1
                        else:
                            ulysses_offset = ulysses_rank * ring_degree * 2
                            forward_rank_idx = ring_rank + ulysses_offset
                            backward_rank_idx = sp_degree - ring_rank - 1 + ulysses_offset

                        # Calculate start and end indices for the forward and backward zigzag
                        start_idx_fwd = forward_rank_idx * chunk_size[i]
                        end_idx_fwd = start_idx_fwd + chunk_size[i]

                        start_idx_bwd = backward_rank_idx * chunk_size[i]
                        end_idx_bwd = start_idx_bwd + chunk_size[i]

                        # Fill new tensors with zigzag data
                        new_attention_mask[i, : chunk_size[i]] = global_attention_mask[i, start_idx_fwd:end_idx_fwd]
                        new_attention_mask[i, chunk_size[i] : 2 * chunk_size[i]] = global_attention_mask[
                            i, start_idx_bwd:end_idx_bwd
                        ]

                        new_position_ids[i, : chunk_size[i]] = global_position_ids[i, start_idx_fwd:end_idx_fwd]
                        new_position_ids[i, chunk_size[i] : 2 * chunk_size[i]] = global_position_ids[
                            i, start_idx_bwd:end_idx_bwd
                        ]

                        new_labels[i, : chunk_size[i]] = global_labels[i, start_idx_fwd:end_idx_fwd]
                        new_labels[i, chunk_size[i] : 2 * chunk_size[i]] = global_labels[i, start_idx_bwd:end_idx_bwd]

                        new_inputs_embeds[i, : chunk_size[i], :] = global_inputs_embeds[i, start_idx_fwd:end_idx_fwd, :]
                        new_inputs_embeds[i, chunk_size[i] : 2 * chunk_size[i], :] = global_inputs_embeds[
                            i, start_idx_bwd:end_idx_bwd, :
                        ]
                else:
                    raise ValueError(f"Invalid ring_type: {ring_type}")
            else:
                global_seq_len = global_attention_mask.shape[-1]
                seq_len_sharded = global_seq_len // sp_degree
                start_idx_reshard = seq_len_sharded * sp_rank
                end_idx_reshard = start_idx_reshard + seq_len_sharded if sp_rank < sp_degree - 1 else global_seq_len

                new_attention_mask = torch.narrow(
                    global_attention_mask, 1, start_idx_reshard, end_idx_reshard - start_idx_reshard
                )
                new_position_ids = torch.narrow(
                    global_position_ids, 1, start_idx_reshard, end_idx_reshard - start_idx_reshard
                )
                new_labels = torch.narrow(global_labels, 1, start_idx_reshard, end_idx_reshard - start_idx_reshard)
                new_inputs_embeds = torch.narrow(
                    global_inputs_embeds, 1, start_idx_reshard, end_idx_reshard - start_idx_reshard
                )

            return (
                None,
                new_position_ids,
                new_attention_mask,
                past_key_values,
                new_inputs_embeds,
                new_labels,
                None,  # sorted_seqlens_in_batch set as None for sequence parallelism
            )

        # kentang-mit@: reorder and repack (reduce computation overhead)
        # requires transformers replacement.
        new_inputs_embeds = []
        new_position_ids = []
        new_labels = []
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        sorted_seqlens_in_batch, sorted_idx = torch.sort(seqlens_in_batch, descending=True)
        max_seqlen = inputs_embeds.shape[1]

        cur_inputs_embeds = []
        cur_position_ids = []
        cur_labels = []
        cur_batch_len = 0
        for i in range(len(sorted_seqlens_in_batch)):
            cur_seqlen = sorted_seqlens_in_batch[i].item()
            if cur_seqlen + cur_batch_len <= max_seqlen:
                cur_batch_len += cur_seqlen
                # each item: num_tokens x num_channels
                # remove padding on-the-fly
                cur_inputs_embeds.append(inputs_embeds[sorted_idx[i]][attention_mask[sorted_idx[i]]])
                cur_position_ids.append(
                    torch.arange(
                        cur_inputs_embeds[-1].shape[0],
                        device=cur_inputs_embeds[-1].device,
                    )
                )
                # each item: num_tokens
                # remove padding on-the-fly
                cur_labels.append(labels[sorted_idx[i]][attention_mask[sorted_idx[i]]])
            else:
                new_inputs_embeds.append(torch.cat(cur_inputs_embeds, 0))
                new_position_ids.append(torch.cat(cur_position_ids, 0))
                new_labels.append(torch.cat(cur_labels, 0))
                # The current batch is too long. We will start a new batch.
                cur_batch_len = cur_seqlen
                cur_inputs_embeds = [inputs_embeds[sorted_idx[i]][attention_mask[sorted_idx[i]]]]
                cur_position_ids = [
                    torch.arange(
                        cur_inputs_embeds[-1].shape[0],
                        device=cur_inputs_embeds[-1].device,
                    )
                ]
                cur_labels = [labels[sorted_idx[i]][attention_mask[sorted_idx[i]]]]
            # Mask the first token in the labels for every sample
            # cur_labels[-1][0] = IGNORE_INDEX

        if len(cur_inputs_embeds):
            new_inputs_embeds.append(torch.cat(cur_inputs_embeds, 0))
            new_position_ids.append(torch.cat(cur_position_ids, 0))
            new_labels.append(torch.cat(cur_labels, 0))

        new_inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            new_inputs_embeds, batch_first=True, padding_value=self.model.config.pad_token_id
        )

        new_position_ids = torch.nn.utils.rnn.pad_sequence(new_position_ids, batch_first=True, padding_value=-1)

        new_labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_INDEX)
        ## yunhao: it's currently a workaround to avoid errors for seq_len < 100
        new_attention_mask = new_position_ids.ne(-1)
        # sanity check
        assert new_attention_mask.sum() == attention_mask.sum()

        return (
            None,
            new_position_ids,
            new_attention_mask,
            past_key_values,
            new_inputs_embeds,
            new_labels,
            sorted_seqlens_in_batch,
        )

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        num_new_tokens = 0
        if model_args.mm_use_patch_start_end:
            num_new_tokens += tokenizer.add_tokens([DEFAULT_PATCH_START_TOKEN, DEFAULT_PATCH_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_video_start_end:
            num_new_tokens += tokenizer.add_tokens([DEFAULT_VIDEO_START_TOKEN, DEFAULT_VIDEO_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens += tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        from quicksviewer.constants import DEFAULT_THUMBNAIL_START_TOKEN, DEFAULT_THUMBNAIL_END_TOKEN
        if model_args.mm_use_thumbnail_start_end:
            num_new_tokens += tokenizer.add_tokens([DEFAULT_THUMBNAIL_START_TOKEN, DEFAULT_THUMBNAIL_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                # assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    # input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                    input_embeddings = embed_tokens_weight
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
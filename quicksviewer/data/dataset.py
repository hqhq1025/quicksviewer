import warnings
import copy
import json
import os
import pickle
import random
import warnings
from io import BytesIO
from typing import Dict
import re
import math
from datetime import datetime

import numpy as np
import torch
import transformers
from PIL import Image
from decord import VideoReader, cpu
from torch.utils.data import Dataset, IterableDataset
from braceexpand import braceexpand
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
import deepspeed.comm as dist

from quicksviewer.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from quicksviewer.data.preprocess import *
from quicksviewer.utils.data_util import TSVReader, uniform_sample, opencv_extract_frames_fps
from quicksviewer.utils.data_util import find_consecutive_segments
from quicksviewer.utils.mm_utils import process_images, get_anyres_image_grid_shape
from quicksviewer.utils.template import random_one_desc
from quicksviewer.train.args import DataArguments, TrainingArguments
from quicksviewer.train.sequence_parallel import (
    extract_local_from_list, extract_local_from_list_by_video,
    extract_local_input_ids, extract_local_input_ids_by_video,
    extract_local_position_ids, extract_local_position_ids_by_video,
    get_pg_manager,
)

random.seed(71)
np.random.seed(71)

class PackedDataset(Dataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
    The dataset also formats the text before tokenization with a specific format that is provided
    by the user.

        Args:
            tokenizer (`transformers.PreTrainedTokenizer`):
                The processor used for processing the data.
    """

    def __init__(
        self,
        dataset,
        batch_size=None,
        infinite=False,
        max_seq_length=1024,
    ):
        self.dataset = dataset
        self.infinite = infinite
        self.gap = batch_size
        self.max_seq_length = max_seq_length

        self.set_epoch(0)

    def _further_split(self, big, lengths, threshold, split, gap):
        def pop(batch, thres):
            now_size = sum([lengths[b] for b in batch])
            for i, b in enumerate(batch):
                now_size -= lengths[b]
                if now_size < thres:
                    return batch[:i+1], batch[i+1:]
        redu = []
        reserved = []

        for b in big:
            redu_part, reserved_part = pop(b, threshold)
            reserved.append(reserved_part)
            redu.extend(redu_part)
        redu = split(redu, gap)
        return redu, reserved

    def _group_samples_by_idxs(self, lengths, gap, max_len_to_avg=1000):
        def split(lst, gap):
            return [lst[i: i + gap] for i in range(0, len(lst), gap)]
        def batch_len(batch, lengths):
            return sum([lengths[i] for i in batch])

        lengths = [x[0] for x in lengths] # x is (length, text/visual)
        indices = list(range(len(lengths)))
        indices = split(indices, gap)

        avg = sum([batch_len(ilst, lengths) for ilst in indices])/len(indices)
        threshold = avg + max_len_to_avg


        big = [batch for batch in indices if batch_len(batch, lengths) >= threshold]
        other = [batch for batch in indices if batch_len(batch, lengths) < threshold]

        # continue split
        iter_idx = 0
        while iter_idx < 10:
            if iter_idx > 5:
                gap = max(1, gap-2)
            iter_idx += 1
            redu, reserved = self._further_split(big, lengths, threshold, split, gap)
            if len(reserved) > 0:
                other += reserved
            big = [batch for batch in redu if batch_len(batch, lengths) >= threshold]
            empty_check = [batch for batch in redu if batch_len(batch, lengths) < threshold]
            if len(empty_check) > 0:
                print(len(empty_check))
                other +=empty_check
            if len(big) == 0:
                break

        batches = other
        return batches

    def set_epoch(self, epoch):
        random.seed(epoch)
        random.shuffle(self.dataset.list_data_dict) # gap is batch size

        lengths = self.dataset.lengths()
        raw_batches = self._group_samples_by_idxs(lengths, gap=self.gap)
        from tqdm import tqdm
        new_batches = []
        for item in tqdm(range(len(raw_batches))):
            idxs = raw_batches[item]
            if len(idxs) != 0:
                new_batches.append(idxs)
        self.batches = new_batches
        # import pdb
        # pdb.set_trace()
        return

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, item):
        idxs = self.batches[item]
        items = [self.dataset[i] for i in idxs]
        return items


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = data #json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.vid_reader = {}
        if self.data_args.video_folder is not None:
            self.vid_reader = TSVReader(self.data_args.video_folder)
        self.text_to_tokens_factor = 1.5
        self.video_to_tokens_factor = 0.5 # Qiji: not exact 1/FPQ, as we consider the workload of ViT
        self.num_per_frame = 64
        self.video_num_frames = data_args.video_num_frames
        self.video_fps = data_args.video_fps
        self.mm_layout_format = getattr(data_args, 'mm_layout_format', 'interleave')
        self.image_aspect_ratio = getattr(data_args, 'image_aspect_ration', 'anyres')
        self.mm_use_thumbnail = getattr(data_args, 'mm_use_thumbnail', False)

        # Prepare sequence parallel environment for loading same samples in a same sq_para group
        self.PROCESS_GROUP_MANAGER = get_pg_manager()
        if self.PROCESS_GROUP_MANAGER is not None:
            self.sp_degree = self.PROCESS_GROUP_MANAGER.sp_degree
            self.sp_rank = self.PROCESS_GROUP_MANAGER.sp_rank
            self.sp_group = self.PROCESS_GROUP_MANAGER.sp_pg
            sequence_parallel_size = data_args.seq_parallel_size
            rank = int(os.environ["RANK"]) // sequence_parallel_size  # RANK is global rank, e.g., 15
            world_size = int(os.environ["WORLD_SIZE"]) // sequence_parallel_size  # WORLD_SIZE is global worldsize, e.g., 16
        else:
            sequence_parallel_size = 1
            rank, world_size = 0, 1
        shared_size = math.ceil(len(self.list_data_dict) / world_size)
        
        self.n_samples = shared_size  # total size
        self.idx_offset = rank * shared_size
        print(f"[Building dataset] ori_n_samples: {len(self.list_data_dict)}, rank: {rank}, world_size: {world_size}, idx_offset: {self.idx_offset}, n_samples: {self.n_samples}")

    def __len__(self):
        return self.n_samples
    
    def get_img_len(self, sample):
        # npatches = 1
        # if self.image_aspect_ratio == 'anyres':
        #     npatches = get_anyres_image_grid_shape(sample['image_size'], self.data_args.image_grid_pinpoints, self.data_args.patch_size)
        # return self.num_per_frame * npatches
        return 256


    def get_vid_len(self, sample):
        vid_len = sample.get("video_len", 0) # video_len for num_frames
        if vid_len == 0:
            if 'finevideo' in sample['video']:
                vid_len = 420
            elif 'activitynet' in sample['video']:
                vid_len = 180
            elif 'sharegpt4video' in sample['video']:
                vid_len = 60
            else:
                vid_len = 30 # default
        vid_len = min(vid_len, self.video_num_frames)
        return vid_len

    @property
    def lengths(self):
        """
        NOTE: compute length of each sample ROUGHLY!
        """
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 256 if 'image' in sample else 0 # rough
            img_tokens = self.num_per_frame + self.get_vid_len(sample)*self.video_to_tokens_factor*self.num_per_frame if 'video' in sample else img_tokens # rough
            if sample['data_type'] == 'conversations':
                text_tokens = sum(len(conv['value'].split()) for conv in sample['conversations'])*self.text_to_tokens_factor # rough
            elif sample['data_type'] == 'interleave':
                text_tokens = sum([len(cap.split()) for cap in sample['caption'] if cap is not None])*self.text_to_tokens_factor # rough
            elif sample['data_type'] == 'caption':
                text_tokens = len(sample['caption'].split())*self.text_to_tokens_factor # rough
            else:
                raise TypeError
            n_len =  text_tokens + img_tokens + 42

            length_list.append(n_len)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if ('image' in sample) or ('video_id' in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        attempt, max_attempt, getitem_succeed = 0, 20, False
        index = i - self.idx_offset
        while attempt < max_attempt:
            try:
                sources = self.list_data_dict[index]
                modality = 'text'
                suffix = None
                if 'image' in sources:
                    modality = 'image'
                    processor = self.data_args.image_processor

                    if sources['data_type'] == 'caption': # Assume one img-caption pair
                        caption = sources['caption']
                        image_file = self.list_data_dict[index]['image']
                        image = Image.open(image_file).convert('RGB')
                        image_sizes = [image.size]
                        image, pathnums_imgs = process_images([image], processor, self.data_args, return_pathnums=True)
                        imidx_in_multi = [0]
                        desc = random_one_desc('image', replace='image')
                        conversations = [{'from':'human', 'value':f'{DEFAULT_IMAGE_TOKEN}{desc}'},{'from':'gpt', 'value':caption}]

                    elif sources['data_type'] == 'interleave': # Assume multi-images
                        image = [Image.open(img).convert('RGB') for img in self.list_data_dict[index]['image'] if img is not None]
                        image_sizes = [img.size for img in image]
                        image, pathnums_imgs = process_images(image, processor, self.data_args, return_pathnums=True)
                        imidx_in_multi = np.cumsum([0] + [image[_].shape[0] for _ in range(len(image)-1)]).tolist()
                        image = torch.cat(image) if isinstance(image, list) else image.view(-1, *image.shape[-3:])
                        # aggregate consecutive imgs or texts
                        conversations = []
                        for i, (cap, img) in enumerate(zip(sources['caption'], sources['image'])):
                            if img is not None:
                                if i>0 and sources['image'][i-1] is not None: # aggregate consecutive imgs
                                    assert conversations[-1]['from'] == 'human'
                                    conversations[-1]['value'] += f'{DEFAULT_IMAGE_TOKEN}'
                                else:
                                    conversations.append({'from':'human', 'value':f'{DEFAULT_IMAGE_TOKEN}'})
                            elif cap is not None:
                                if i>0 and sources['caption'][i-1] is not None: # aggregate consecutive txts
                                    assert conversations[-1]['from'] == 'gpt'
                                    conversations[-1]['value'] += ' ' + cap
                                else:
                                    conversations.append({'from':'gpt', 'value':cap})
                        if conversations[0]['from'] == 'gpt': # must begin with img
                            if len(conversations)>2:
                                remove_imgs = len(re.findall(DEFAULT_IMAGE_TOKEN, conversations[0]['value']))
                                conversations = conversations[1:]
                                image = image[sum(pathnums_imgs[:remove_imgs]):]
                                pathnums_imgs = pathnums_imgs[remove_imgs:]
                            else:
                                conversations = conversations[::-1]

                        if conversations[-1]['from'] == 'human': # must end with txt
                            if len(conversations)>2:
                                remove_imgs = len(re.findall(DEFAULT_IMAGE_TOKEN, conversations[-1]['value']))
                                conversations = conversations[:-1]
                                image = image[:-sum(pathnums_imgs[-remove_imgs:])]
                                pathnums_imgs = pathnums_imgs[:-remove_imgs]
                            else:
                                conversations+[{'from':'gpt', 'value':f'an image.'}]
                        

                    elif sources['data_type'] == 'conversations': # Support multi-images for this data_type
                        image_files = self.list_data_dict[index]['image']
                        if isinstance(image_files, str):
                            image_files = [image_files]
                        images = [Image.open(_f).convert('RGB') for _f in image_files]
                        image_sizes = [_i.size for _i in images]
                        image, pathnums_imgs = process_images(images, processor, self.data_args, return_pathnums=True)
                        imidx_in_multi = np.cumsum([len(_) for _ in image])-len(image[0])
                        conversations = sources["conversations"]
                        imnum_inconv = sum([len(re.findall('<image>',conversations[_]['value'])) for _ in range(0,len(conversations),2)])
                        if imnum_inconv != len(image):
                            conversations[0]["value"] = "<image>"*len(image) + '\n' + conversations[0]["value"]

                    sources = preprocess_multimodal_image(
                        copy.deepcopy([conversations]),
                        self.data_args,
                        pathnums_imgs=pathnums_imgs
                        )

                elif 'video_id' in sources or 'video' in sources:
                    modality = 'video'
                    data_type = sources.get('data_type', 'conversations')
                    if data_type == 'caption':
                        caption = sources['caption']
                        desc = random_one_desc('video', replace='video')
                        conversations = [{'from':'human', 'value':f'{DEFAULT_IMAGE_TOKEN}{desc}'},{'from':'gpt', 'value':caption}]
                        video_file = sources.get('video', sources.get('video_id'))
                        start_sec, end_sec = sources.get('start_sec',None), sources.get('end_sec',None)
                        _timestart = datetime.now()
                        video_bs = self.vid_reader.get(video_file)
                        if video_bs is not None:
                            video, indices = uniform_sample(video_bs, min(self.video_num_frames, len(video_bs)))
                            timestamps = [_t for _i,_t in enumerate(sources.get('timestamps',[])) if _i in indices]
                            if len(timestamps) == 0:
                                timestamps = [round(float(_t),2) for _i,_t in enumerate(uniform_sample(range(round(end_sec - start_sec)), len(video_bs))[0]) if _i in indices] # Default 1FPS
                            video = [Image.open(BytesIO(img)).convert('RGB') for img in video]
                        else: # load directly
                            video, timestamps = opencv_extract_frames_fps(video_file, self.video_num_frames, self.video_fps,start_sec=start_sec, end_sec=end_sec)
                    elif data_type == 'interleave':
                        pass
                    elif data_type == 'conversations':
                        conversations = sources["conversations"]
                        for conv in conversations:
                            conv["value"] = conv["value"].replace("<video>", "<image>")
                        if '<image>' not in conversations[0]["value"]:
                            conversations[0]["value"] =  "<image>\n" + conversations[0]["value"]
                        video_file = sources.get('video', sources.get('video_id'))
                        start_sec, end_sec = sources.get('start_sec',None), sources.get('end_sec',None)
                        video_bs = self.vid_reader.get(video_file)
                        if video_bs is not None:
                            start_sec, end_sec = sources.get('start_sec',0), sources.get('end_sec',len(video_bs)) # Default 1FPS
                            video, indices = uniform_sample(video_bs, min(self.video_num_frames, len(video_bs)))
                            timestamps = [_t for _i,_t in enumerate(sources.get('timestamps',[])) if _i in indices]
                            if len(timestamps) == 0:
                                timestamps = [round(float(_t),2) for _i,_t in enumerate(uniform_sample(range(round(end_sec - start_sec)), len(video_bs))[0]) if _i in indices] # Default 1FPS
                            video = [Image.open(BytesIO(img)).convert('RGB') for img in video_bs]
                        else: # load directly
                            video, timestamps = opencv_extract_frames_fps(video_file, self.video_num_frames, self.video_fps,start_sec=start_sec, end_sec=end_sec)
                    if len(video) < 2: # Qiji: hard rule to restrict video duration
                        video, timestamps = opencv_extract_frames_fps(video_file, self.video_num_frames, self.video_fps+1,start_sec=start_sec, end_sec=end_sec)
                    image_sizes = [img.size for img in video]
                    processor = self.data_args.image_processor
                    image = process_images(video, processor, self.data_args, image_aspect_ratio='original')
                    imidx_in_multi = list(range(len(video)))
                    assert len(video) == len(image)
                    image = image.view(-1, *image.shape[-3:])

                    sources = preprocess_multimodal_video(
                        copy.deepcopy([conversations]),
                        self.data_args,
                        nframes=len(video),
                        frame_timestamps=timestamps,
                        is_cubing=self.data_args.is_cubing,
                        add_thumbnail=self.mm_use_thumbnail,
                        )
                else:
                    sources = copy.deepcopy([sources["conversations"]])
                getitem_succeed = True
                break
            except:
                import traceback
                traceback.print_exc()
                attempt += 1
                print(
                    f"Error in loading {index}, {self.list_data_dict[index].get('image', self.list_data_dict[index].get('video', 'no visual input'))}, retrying...")
                if attempt == max_attempt // 2:
                    index = i + 1 - self.idx_offset
                    print(f"#### Warning #### Re-specifying data sample with index: {index}, rank: {os.environ['RANK']}")
                    # check if all indexs in a same sp group are consistent
        if not getitem_succeed:
            raise IOError(f"Failed in loading {index}, instance: {self.list_data_dict[index]}")

        has_image = ('image' in self.list_data_dict[index]) or ('video' in self.list_data_dict[index]) or('video_id' in self.list_data_dict[index])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=has_image,
            prompt=self.data_args.input_prompt,
            refine_prompt=self.data_args.refine_prompt)

        if 'prompt' in data_dict:
            prompt = data_dict['prompt']
        else:
            prompt = None

        if isinstance(index, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        if has_image:
            if isinstance(image, torch.Tensor) and image.ndim>4:
                image = image.view(-1, *image.shape[-3:])
            elif isinstance(image, list) and image[0].ndim == 4:
                image = torch.cat(image, dim=0)

        if any([k in self.list_data_dict[index] for k in ['image', 'video', 'video_id']]):
            data_dict['image'] = image
            data_dict['image_sizes'] = image_sizes
            data_dict['imidx_in_multi'] = imidx_in_multi
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            if hasattr(self.data_args.image_processor, "crop_size"):
                crop_size = self.data_args.image_processor.crop_size
            else:
                crop_size = self.data_args.image_processor.size
            patch_size = self.data_args.patch_size
            data_dict['image'] = torch.zeros(1, 3, crop_size['height'], crop_size['width'])
            data_dict['image_sizes'] = [[crop_size['height'], crop_size['width']]]
            data_dict['imidx_in_multi'] = [0]
            # dummy (for SP partition smoothly, comment below 3 lines when no needing SP)
            data_dict['input_ids'] = torch.cat([data_dict['input_ids'], torch.full([1,], IMAGE_TOKEN_INDEX)])
            data_dict['labels'] = torch.cat([data_dict['labels'], torch.full([1,], IGNORE_INDEX)])
            modality = 'image'
            if len(data_dict['input_ids']) <= 1:
                print(f"#### WARNING: not effective inputs #### sources: {sources}")

        # prompt exist in the data
        if prompt is not None:
            data_dict['prompt'] = prompt
        data_dict['modality'] = modality

        # Other information for easy evaluation
        if 'metadata' in self.list_data_dict[index]:
            data_dict['metadata'] = self.list_data_dict[index]['metadata']

        if len(data_dict['input_ids']) < 1:
            print(f"!!!!!!!!\n The input_ids is empty, using dummy instead.\n Souces: {sources}\n self.list_data_dict[index]: {self.list_data_dict[index]}")
            crop_size = self.data_args.image_processor.crop_size if hasattr(self.data_args.image_processor, "crop_size") else  self.data_args.image_processor.size
            data_dict['image'] = torch.zeros(1, 3, crop_size['height'], crop_size['width'])
            data_dict['input_ids'] = torch.tensor([IMAGE_TOKEN_INDEX, 0], dtype=torch.long)
            data_dict['labels'] = torch.tensor([IGNORE_INDEX, IGNORE_INDEX], dtype=torch.long)

        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # instances = instances[0]

        input_ids, labels = tuple([torch.LongTensor(instance[key]) for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                        batch_first=True,
                                                        padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        seq_lens = [len(inp) for inp in input_ids]
        modalities = [instance['modality'] for instance in instances]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            modalities = modalities,
            metadata = [inst['metadata'] for inst in instances if 'metadata' in inst],
            # seq_lens=seq_lens,
            need_repack = True
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images) and len(images) > 1:
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

            batch['image_sizes'] = [instance['image_sizes'] for instance in instances]
            batch['imidx_in_multi'] = [instance['imidx_in_multi'] for instance in instances]

        if 'prompt' in instances[0]:
            batch['prompts'] = [instance['prompt'] for instance in instances]

        return batch
    
@dataclass
class DataCollatorForPackingSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer, data_args):
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([torch.LongTensor(instance[key]) for instance in instances]
                                  for key in ("input_ids", "labels"))
        modalities = [instance['modality'] for instance in instances]

        images, image_sizes, imidx_in_multi = [], [], []
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images) and len(images) > 1:
                images = torch.stack(images)
            else:
                images = images

        # Packing
        NUM_TOKENS_PER_IMAGE = 64
        combined = sorted(
            zip(input_ids, labels, images, modalities),
            key=lambda x: len(x[2]) * (NUM_TOKENS_PER_IMAGE - 1) + x[0].size(-1),
            reverse=True,  # Start Packing from the sequence with most images.
        )
        sorted_ids, sorted_labels, sorted_images, sorted_modalities = zip(*combined)
        sorted_ids, sorted_labels, sorted_images, sorted_modalities = list(sorted_ids), list(sorted_labels), list(sorted_images), list(sorted_modalities)
        max_sample_len = 0
        MAX_SAMPLE_LEN = 32768
        MAX_SEQ_LEN = 16384

        batches = []
        label_batches = []
        position_ids = [] # Qiji: Need to handle cubing-merge during training/inference
        batch_images = []
        seqlens_in_batch = [] # Qiji: Need to handle cubing-merge during training/inference
        batch_modalities = []
        batch_videos_bound = []

        max_num_images = max([len(_images) for _images in images])
        i = 0
        while i < len(sorted_ids):
            max_seq_length = MAX_SEQ_LEN # Qiji: fixed 
            current_batch = torch.tensor([], dtype=torch.int32)
            current_label_batch = torch.tensor([], dtype=torch.int32)
            current_position_ids = torch.tensor([], dtype=torch.int32)
            current_batch_images = []
            current_num_images = 0
            current_len = 0
            current_modalities = []
            current_video_bound = []

            # Pack a few samples into one sample
            vbound_prev = 0
            while i < len(sorted_ids):
                num_images = (sorted_ids[i] == IMAGE_TOKEN_INDEX).sum().item()
                num_image_tokens_added = num_images * (NUM_TOKENS_PER_IMAGE - 1) # Qiji: to remove the placeholders
                num_incoming_tokens = sorted_ids[i].size(-1) + num_image_tokens_added

                if num_incoming_tokens > max_seq_length:
                    # Add Dummy sample to avoid empty batch
                    print(f"Warning: Dummying one packed sample as it has {num_incoming_tokens} tokens, which is greater than max seq len {max_seq_length}.")
                    if hasattr(self.data_args.image_processor, "crop_size"):
                        crop_size = self.data_args.image_processor.crop_size
                    else:
                        crop_size = self.data_args.image_processor.size
                    sorted_images[i] = torch.zeros(1, 3, crop_size['height'], crop_size['width'])
                    sorted_ids[i] = torch.full([1,], IMAGE_TOKEN_INDEX)
                    sorted_labels[i] = torch.full([1,], IGNORE_INDEX)
                    sorted_modalities[i] = 'image'
                    num_incoming_tokens = sorted_ids[i].size(-1) + NUM_TOKENS_PER_IMAGE - 1
                    num_images = 1

                if (
                    current_len + num_incoming_tokens <= MAX_SAMPLE_LEN
                ) and (current_len + num_incoming_tokens <= max_seq_length):
                    current_num_images += num_images
                    current_len += num_incoming_tokens
                    current_position_ids = torch.cat(
                        (current_position_ids, torch.arange(start=0, end=num_incoming_tokens)), dim=0
                    )
                    current_batch = torch.cat((current_batch, sorted_ids[i]), dim=0)
                    current_label_batch = torch.cat((current_label_batch, sorted_labels[i]), dim=0)
                    seqlens_in_batch.append(num_incoming_tokens)
                    current_batch_images.extend(sorted_images[i])
                    current_modalities.extend([sorted_modalities[i]]*len(sorted_images[i]))
                    if sorted_modalities[i] == 'video':
                        current_video_bound.append([vbound_prev, vbound_prev+len(sorted_images[i])])
                    vbound_prev += len(sorted_images[i])
                    i += 1
                else:
                    break

            max_sample_len = max(max_sample_len, current_len)
            batches.append(current_batch)
            label_batches.append(current_label_batch)
            position_ids.append(current_position_ids)
            batch_images.append(torch.stack(current_batch_images))
            batch_modalities.append(current_modalities)
            if current_video_bound != []:
                batch_videos_bound.append(current_video_bound)

            try:
                assert current_num_images == len(torch.where(current_batch == IMAGE_TOKEN_INDEX)[0].tolist())
            except AssertionError:
                print(
                    f"Error len(torch.where(batches[i] == IMAGE_TOKEN_INDEX)[0].tolist():",
                    len(torch.where(current_batch == IMAGE_TOKEN_INDEX)[0].tolist()),
                )
                raise AssertionError

        # Padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            batches, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(label_batches, batch_first=True, padding_value=IGNORE_INDEX)
        seqlens_in_batch = torch.stack([torch.tensor(x) for x in seqlens_in_batch], axis=0).flatten()
        position_ids = torch.nn.utils.rnn.pad_sequence(position_ids, batch_first=True, padding_value=-1)


        # cube specific
        image_sizes = [instance['image_sizes'] for instance in instances]
        imidx_in_multi = [instance['imidx_in_multi'] for instance in instances]
        batch_videos_bound = [vb for bvb in batch_videos_bound for vb in bvb]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            # notice that we inject attention mask here
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            seqlens_in_batch=seqlens_in_batch,
            images=batch_images,
            position_ids=position_ids,
            modalities=batch_modalities, # it's chunked so not easy to decide modalities
            image_sizes=image_sizes,
            imidx_in_multi=imidx_in_multi,
            videos_bound=batch_videos_bound,
            metadata = [inst['metadata'] for inst in instances if 'metadata' in inst],
            need_repack = False
        )

        return batch
    

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, training_args, list_data_dict=None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if list_data_dict is None:
        imdata_list, viddata_list = [], []
        for path in data_args.data_path.split(';'):
            for p in braceexpand(path.strip()):
                p, num_times = p.split('*') if '*' in p else (p, 1)
                p, num_range = p.split('#') if '#' in p else (p, None)
                num_times = int(num_times)
                num_range = [int(x) for x in num_range.split('_')] if num_range is not None else [0,None] # Specify chunk to train
                if "ego4d_1fps512frames_" in p:
                    viddata_list.extend(([{k:v if k!='video' else p.split('/')[-2]+'/'+v for k,v in json.loads(line).items()} for line in open(p)]*num_times)[num_range[0]:num_range[1]])
                elif 'video_data' in p or 'tsv_data' in p:
                    if p.endswith('json'): viddata_list.extend((json.load(open(p))*num_times)[num_range[0]:num_range[1]])
                    elif p.endswith('jsonl'): viddata_list.extend(([json.loads(line) for line in open(p)]*num_times)[num_range[0]:num_range[1]])
                else:
                    if p.endswith('json'): imdata_list.extend((json.load(open(p))*num_times)[num_range[0]:num_range[1]])
                    elif p.endswith('jsonl'): imdata_list.extend(([json.loads(line) for line in open(p)]*num_times)[num_range[0]:num_range[1]])
        list_data_dict = imdata_list + viddata_list
        if len(imdata_list)>0 and len(viddata_list)>0:
            # shuffle and evenly insert longvideo between images
            random.shuffle(imdata_list); random.shuffle(viddata_list)
            _i_tot, _i_added, _v_tot, _v_added = len(imdata_list), 0, len(viddata_list), 0
            data_list = []
            while len(data_list) < _i_tot+_v_tot:
                gap_i2v, gap_v2i = math.ceil((_i_tot-_i_added)/(_v_tot-_v_added)), math.ceil((_v_tot-_v_added)/(_i_tot-_i_added))
                data_list.extend(imdata_list[_i_added: _i_added+gap_i2v]); _i_added+=gap_i2v
                data_list.extend(viddata_list[_v_added: _v_added+gap_v2i]); _v_added+=gap_v2i
            print(f"Total image-samples: {len(imdata_list)}, total video-samples: {len(viddata_list)}")
            list_data_dict = data_list
    
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                            data=list_data_dict,
                                            data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)



@dataclass
class DataCollatorForSupervisedDatasetSeqParallel:
    """Collate examples for supervised fine-tuning.
    This class is originally implemented by the LLaVA team and
    modified by Haotian Tang."""

    tokenizer: transformers.PreTrainedTokenizer
    data_args: DataArguments
    training_args: TrainingArguments
    sp_degree: int
    sp_rank: int
    ring_degree: int
    ring_type: str
    is_cubing: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, images, modalities = [], [], [], []
        for instance in instances:
            if not isinstance(instance["input_ids"], list):
                input_ids.append(instance["input_ids"])
            else:
                input_ids += instance["input_ids"]
            if not isinstance(instance["labels"], list):
                labels.append(instance["labels"])
            else:
                labels += instance["labels"]
            # Note (kentang-mit@: we do not directly push tensors to
            # images, but list of tensors.
            if instance["image"] is not None:
                cur_image = instance["image"]
                assert len(cur_image.shape) == 4
                # n_images, 3, size, size
                if cur_image.shape[0] == 0:
                    warnings.warn("loaded one sample without images.")
                if not isinstance(instance["input_ids"], list):
                    # datasets other than coyo, not packing >1 samples together
                    images.append(cur_image)
                else:
                    # coyo-like datasets
                    print(f"Extending: {cur_image.chunk(cur_image.size(0), 0).shape}")
                    images.extend(cur_image.chunk(cur_image.size(0), 0))
            else:
                warnings.warn("loaded one sample without images.")
                images.append([])
            modalities.append(instance["modality"])
        # kentang-mit@: we need to make sure these two lists have
        # the same length. We will use input_ids to filter out images corresponding
        # to truncated <image> tokens later.
        max_num_images = max([len(_images) for _images in images])
        for _images, _input_ids, _moda in zip(images, input_ids, modalities):
            n_images = len(_images)+1 if (self.is_cubing and _moda=='video' and self.data_args.mm_use_thumbnail) else len(_images) # Qiji: Add thumbnail video placeholdered in input_ids
            assert (
                n_images == (_input_ids == IMAGE_TOKEN_INDEX).sum().item()
            ), f"Number mismatch between images and placeholder image tokens in 'len(_images) == (_input_ids == IMAGE_TOKEN_INDEX).sum().item()'.\
                Expect to have {len(_images)} images but only found {(_input_ids == IMAGE_TOKEN_INDEX).sum().item()} images in tokens. \
                Error input_ids: {_input_ids} {self.tokenizer.decode([x if x != -200 else 200 for x in _input_ids])}"

        # TODO: Remove the hard coding of NUM_TOKENS_PER_IMAGE
        NUM_TOKENS_PER_IMAGE = 64
        if hasattr(self.data_args.image_processor, "crop_size"):
            crop_size = self.data_args.image_processor.crop_size
        else:
            crop_size = self.data_args.image_processor.size

        # Init the padding sample
        seq_id = 0
        while seq_id < len(input_ids):
            # Skip the samples without images
            dummy_image = torch.ones((1, 3, crop_size["height"], crop_size["width"]), device=input_ids[seq_id].device)
            dummy_input_ids = torch.zeros_like(input_ids[seq_id][:1])
            dummy_input_ids[0] = IMAGE_TOKEN_INDEX
            dummy_labels = copy.deepcopy(dummy_input_ids)
            dummy_labels[0] = IGNORE_INDEX
            dummy_seqlen = NUM_TOKENS_PER_IMAGE  # TODO: Check the hard coding of 2
            dummy_position_ids = torch.arange(start=0, end=dummy_seqlen, dtype=torch.int32)
            break

        # Sort with the real length of the sequence
        combined = sorted(
            zip(input_ids, labels, images, modalities),
            key=lambda x: len(x[2]) * (NUM_TOKENS_PER_IMAGE - 1) + x[0].size(-1),
            reverse=True,  # Start Packing from the sequence with most images.
        )
        sorted_ids, sorted_labels, sorted_images, sorted_modalities = zip(*combined)
        sorted_ids, sorted_labels, sorted_images, sorted_modalities = list(sorted_ids), list(sorted_labels), list(sorted_images), list(sorted_modalities)
        # max_seq_length = self.tokenizer.model_max_length  # len(sorted_ids[0])
        max_sample_len = 0

        batches = []
        label_batches = []
        position_ids = [] # Qiji: Need to handle cubing-merge during training/inference
        batch_images = []
        seqlens_in_batch = [] # Qiji: Need to handle cubing-merge during training/inference
        batch_modalities = []
        batch_videos_bound = []

        i = 0
        while i < len(sorted_ids):
            # if not self.is_cubing or sorted_modalities[i]!='video':
            max_seq_length = self.tokenizer.model_max_length # Qiji: fixed max_seq_length
            current_batch = torch.tensor([], dtype=torch.int32)
            current_label_batch = torch.tensor([], dtype=torch.int32)
            current_position_ids = torch.tensor([], dtype=torch.int32)
            current_batch_images = []
            current_num_images = 0
            current_len = 0
            current_num_samples = 0
            current_modalities = []
            current_video_bound = []

            # Pack a few samples into one sample
            vbound_prev = 0
            while i < len(sorted_ids):
                num_images = (sorted_ids[i] == IMAGE_TOKEN_INDEX).sum().item()
                num_image_tokens_added = num_images * (NUM_TOKENS_PER_IMAGE - 1) # Qiji: to 
                num_incoming_tokens = sorted_ids[i].size(-1) + num_image_tokens_added

                # Handle RingAttn_Varlen which requires `seqlens_in_batch` should be divisible by `ring_degree`
                if self.ring_degree > 1:
                    RING_PAD_TOKEN_INDEX = 2
                    if self.ring_type == "ring_varlen":
                        if num_incoming_tokens % self.sp_degree != 0:
                            pad_len = self.sp_degree - num_incoming_tokens % self.sp_degree
                            num_incoming_tokens += pad_len
                            # pad `input_ids`
                            pad_tensor = torch.full(
                                (pad_len,), RING_PAD_TOKEN_INDEX, dtype=sorted_ids[i].dtype, device=sorted_ids[i].device
                            )
                            sorted_ids[i] = torch.cat([sorted_ids[i], pad_tensor])

                            # pad `label`
                            pad_label_tensor = torch.full(
                                (pad_len,), IGNORE_INDEX, dtype=sorted_labels[i].dtype, device=sorted_labels[i].device
                            )
                            sorted_labels[i] = torch.cat([sorted_labels[i], pad_label_tensor])
                    elif self.ring_type == "zigzag_ring_varlen":
                        self.zigzag_sp_degree = self.sp_degree * 2
                        if num_incoming_tokens % self.zigzag_sp_degree != 0:
                            pad_len = self.zigzag_sp_degree - num_incoming_tokens % self.zigzag_sp_degree
                            num_incoming_tokens += pad_len
                            # pad `input_ids`
                            pad_tensor = torch.full(
                                (pad_len,), RING_PAD_TOKEN_INDEX, dtype=sorted_ids[i].dtype, device=sorted_ids[i].device
                            )
                            sorted_ids[i] = torch.cat([sorted_ids[i], pad_tensor])

                            # pad `label`
                            pad_label_tensor = torch.full(
                                (pad_len,), IGNORE_INDEX, dtype=sorted_labels[i].dtype, device=sorted_labels[i].device
                            )
                            sorted_labels[i] = torch.cat([sorted_labels[i], pad_label_tensor])
                    else:
                        raise ValueError(f"Invalid ring_type: {self.ring_type}")

                if num_incoming_tokens > max_seq_length:
                    print(
                        f"Warning: Skipping one packed sample with {num_incoming_tokens} tokens,\
                        please consider increase max seq len {max_seq_length}."
                    )
                    i += 1
                    continue

                if (
                    (current_num_images == 0)
                    or (current_num_images < self.sp_degree)
                    or (
                        (current_num_images + num_images <= max_num_images)
                        and (current_len + num_incoming_tokens <= max_sample_len)
                    )
                ) and (current_len + num_incoming_tokens <= max_seq_length) \
                  and (sorted_modalities[i]!='video' or 'video' not in current_modalities): # Qiji: ensure videos cannot be packed into one sequence !!
                    current_num_images += num_images
                    current_len += num_incoming_tokens
                    current_num_samples += 1
                    current_position_ids = torch.cat(
                        (current_position_ids, torch.arange(start=0, end=num_incoming_tokens)), dim=0
                    )
                    current_batch = torch.cat((current_batch, sorted_ids[i]), dim=0)
                    sorted_labels[i][0] = IGNORE_INDEX
                    current_label_batch = torch.cat((current_label_batch, sorted_labels[i]), dim=0)
                    seqlens_in_batch.append(num_incoming_tokens)
                    current_batch_images.extend(sorted_images[i])
                    current_modalities.extend([sorted_modalities[i]]*len(sorted_images[i]))
                    if sorted_modalities[i] == 'video':
                        current_video_bound = [vbound_prev, vbound_prev+len(sorted_images[i])]
                    vbound_prev += len(sorted_images[i])
                    i += 1
                    # assert current_num_images == len(current_batch_images)
                else:
                    break

            # Padding the sample with the dummy image sample, if there are no enough images
            MAX_RETRY = self.sp_degree
            num_retry = 0
            while current_num_images < self.sp_degree and current_len < max_seq_length and num_retry <= MAX_RETRY:
                current_num_images += dummy_image.size(0)
                current_len += dummy_seqlen
                current_num_samples += 1
                current_position_ids = torch.cat((current_position_ids, dummy_position_ids), dim=0)
                current_batch = torch.cat((current_batch, dummy_input_ids), dim=0)
                current_label_batch = torch.cat((current_label_batch, dummy_labels), dim=0)
                seqlens_in_batch.append(dummy_seqlen)
                current_batch_images.extend(dummy_image)
                current_modalities.append('image')
                num_retry += 1

            # Drop the samples that do not have enough images
            if current_num_images < self.sp_degree:
                print(f"Warning: Skipping one packed sample with {current_num_images} images")
                seqlens_in_batch = seqlens_in_batch[:-current_num_samples]
                continue

            max_sample_len = max(max_sample_len, current_len)
            batches.append(current_batch)
            label_batches.append(current_label_batch)
            position_ids.append(current_position_ids)
            batch_images.append(current_batch_images)
            batch_modalities.append(current_modalities)
            batch_videos_bound.append(current_video_bound)

            try:
                assert current_num_images == len(torch.where(current_batch == IMAGE_TOKEN_INDEX)[0].tolist())
            except AssertionError:
                print(f"Error num_images on {self.sp_rank}", current_num_images)
                print("current_batch", current_batch)
                print(
                    f"Error len(torch.where(batches[i] == IMAGE_TOKEN_INDEX)[0].tolist() on {self.sp_rank}:",
                    len(torch.where(current_batch == IMAGE_TOKEN_INDEX)[0].tolist()),
                )
                print(f"Error len(current_batch_images) on {self.sp_rank}:", len(current_batch_images))
                raise AssertionError

        # Split for sequence parallelism
        ori_batch_videos_bound = copy.copy(batch_videos_bound)
        batch_num_images = 0
        for i in range(len(batches)):
            image_token_indices = torch.where(batches[i] == IMAGE_TOKEN_INDEX)[0].tolist()
            image_ids = torch.arange(0, len(image_token_indices), dtype=torch.int32)
            if self.data_args.mm_use_thumbnail and 'video' in batch_modalities[i]: # Qiji: for consistent split between images & input_ids
                video_bound_thumb = [ori_batch_videos_bound[i][0], ori_batch_videos_bound[i][1]+1]
                batches[i] = extract_local_input_ids_by_video(
                    batches[i], image_token_indices, self.sp_rank, self.sp_degree, self.tokenizer.bos_token_id, video_bound = video_bound_thumb,
                )
                label_batches[i] = extract_local_input_ids_by_video(
                    label_batches[i], image_token_indices, self.sp_rank, self.sp_degree, self.tokenizer.bos_token_id, video_bound = video_bound_thumb,
                )

                images_i = extract_local_from_list_by_video([dummy_image]+batch_images[i], self.sp_rank, self.sp_degree, video_bound_thumb)
                images_i = images_i[int(self.sp_rank==0):] # Qiji: remove dummy in first chunk
                batch_images[i] = torch.concat(images_i, dim=0)
                batch_modalities[i] = extract_local_from_list_by_video(['video']+batch_modalities[i], self.sp_rank, self.sp_degree, video_bound_thumb)
                batch_modalities[i] = batch_modalities[i][int(self.sp_rank==0):]

                position_ids[i] = extract_local_position_ids_by_video(
                    position_ids[i], image_token_indices, image_ids, self.sp_rank, self.sp_degree, NUM_TOKENS_PER_IMAGE - 1, video_bound_thumb
                )
            else:
                batches[i] = extract_local_input_ids(
                    batches[i], image_token_indices, self.sp_rank, self.sp_degree, self.tokenizer.bos_token_id
                )
                label_batches[i] = extract_local_input_ids(
                    label_batches[i], image_token_indices, self.sp_rank, self.sp_degree, self.tokenizer.bos_token_id
                )
                
                batch_images[i] = torch.concat(
                    extract_local_from_list(batch_images[i], self.sp_rank, self.sp_degree), dim=0
                )
                batch_modalities[i] = extract_local_from_list(batch_modalities[i], self.sp_rank, self.sp_degree)

                position_ids[i] = extract_local_position_ids(
                    position_ids[i], image_token_indices, image_ids, self.sp_rank, self.sp_degree, NUM_TOKENS_PER_IMAGE - 1
                )
            H, W = batch_images[i].size(-2), batch_images[i].size(-1)
            batch_images[i] = batch_images[i].reshape(-1, 3, W, H)
            num_images = len(batch_images[i])

            _vbound = find_consecutive_segments(batch_modalities[i], label='video')
            batch_videos_bound[i] = [[_b+batch_num_images for _b in _vbound[0]]] if len(_vbound)>0 else []
            for _i, _vb in enumerate(_vbound[1:], 1):
                batch_videos_bound[i].append([_b+len(_vbound[_i-1])+batch_num_images for _b in _vb])
            batch_num_images += num_images
            try:
                n_imgs_in_vbound = sum([_b[1]-_b[0] for _b in batch_videos_bound[i] if len(_b)>0]) if 'video' in batch_modalities[i] else 0
                assert n_imgs_in_vbound + sum([m=='image' for m in batch_modalities[i]]) == num_images
            except AssertionError:
                print(f"Error in building videos bound\n sp_rank: {self.sp_rank}\n batch_videos_bound_i: {batch_videos_bound[i]}\n num_images: {num_images}\n num_imgs_in_inp_i: {(batches[i]==-200).sum()}\n batch_modalities[i]: {batch_modalities[i]}")
                raise AssertionError
            
        input_ids = torch.nn.utils.rnn.pad_sequence(
            batches, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(label_batches, batch_first=True, padding_value=IGNORE_INDEX)
        seqlens_in_batch = [torch.tensor(x) for x in seqlens_in_batch]
        seqlens_in_batch = torch.stack(seqlens_in_batch, axis=0)
        seqlens_in_batch = seqlens_in_batch.flatten()
        # print(f"seqlens_in_batch: {seqlens_in_batch}")
        position_ids = torch.nn.utils.rnn.pad_sequence(position_ids, batch_first=True, padding_value=-1)

        if batch_images:
            flat_batch_images = torch.concat(batch_images, dim=0)
        else:
            flat_batch_images = None

        # cube specific
        image_sizes = [instance['image_sizes'] for instance in instances]
        imidx_in_multi = [instance['imidx_in_multi'] for instance in instances]
        batch_videos_bound = [vb for bvb in batch_videos_bound for vb in bvb] # flat

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            # notice that we inject attention mask here
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            seqlens_in_batch=seqlens_in_batch,
            # images=flat_batch_images,
            images=batch_images,
            position_ids=position_ids,
            modalities=batch_modalities, # it's chunked so not easy to decide modalities
            image_sizes=image_sizes,
            imidx_in_multi=imidx_in_multi,
            videos_bound=batch_videos_bound if len(batch_videos_bound)!=0 else None,
            metadata = [inst['metadata'] for inst in instances if 'metadata' in inst],
            need_repack = True
        )

        return batch
    
def make_supervised_data_module_sp(
    tokenizer: PreTrainedTokenizer,
    data_args,
    training_args,
    list_data_dict=None
) -> Dict:
    if list_data_dict is None:
        imdata_list, viddata_list = [], []
        for path in data_args.data_path.split(';'):
            for p in braceexpand(path.strip()):
                p, num_times = p.split('*') if '*' in p else (p, 1)
                p, num_range = p.split('#') if '#' in p else (p, None)
                num_times = int(num_times)
                num_range = [int(x) for x in num_range.split('_')] if num_range is not None else [0,None] # Specify chunk to train
                if "ego4d_1fps512frames_" in p:
                    viddata_list.extend(([{k:v if k!='video' else p.split('/')[-2]+'/'+v for k,v in json.loads(line).items()} for line in open(p)]*num_times)[num_range[0]:num_range[1]])
                elif 'video_data' in p or 'tsv_data' in p:
                    if p.endswith('json'): viddata_list.extend((json.load(open(p))*num_times)[num_range[0]:num_range[1]])
                    elif p.endswith('jsonl'): viddata_list.extend(([json.loads(line) for line in open(p)]*num_times)[num_range[0]:num_range[1]])
                else:
                    if p.endswith('json'): imdata_list.extend((json.load(open(p))*num_times)[num_range[0]:num_range[1]])
                    elif p.endswith('jsonl'): imdata_list.extend(([json.loads(line) for line in open(p)]*num_times)[num_range[0]:num_range[1]])
        list_data_dict = imdata_list + viddata_list
        if len(imdata_list)>0 and len(viddata_list)>0:
            # shuffle and evenly insert longvideo between images
            random.shuffle(imdata_list); random.shuffle(viddata_list)
            _i_tot, _i_added, _v_tot, _v_added = len(imdata_list), 0, len(viddata_list), 0
            data_list = []
            while len(data_list) < _i_tot+_v_tot:
                gap_i2v, gap_v2i = math.ceil((_i_tot-_i_added)/(_v_tot-_v_added)), math.ceil((_v_tot-_v_added)/(_i_tot-_i_added))
                data_list.extend(imdata_list[_i_added: _i_added+gap_i2v]); _i_added+=gap_i2v
                data_list.extend(viddata_list[_v_added: _v_added+gap_v2i]); _v_added+=gap_v2i
            print(f"Total image-samples: {len(imdata_list)}, total video-samples: {len(viddata_list)}")
            list_data_dict = data_list
    data_args.seq_parallel_size = training_args.seq_parallel_size
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                            data=list_data_dict,
                                            data_args=data_args)
    
    PROCESS_GROUP_MANAGER = get_pg_manager()
    sp_degree = training_args.seq_parallel_size
    sp_rank = PROCESS_GROUP_MANAGER.sp_rank
    ring_degree = PROCESS_GROUP_MANAGER.ring_degree
    ring_type = PROCESS_GROUP_MANAGER.ring_type
    data_collator = DataCollatorForSupervisedDatasetSeqParallel(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
        sp_degree=sp_degree,
        sp_rank=sp_rank,
        ring_degree=ring_degree,
        ring_type=ring_type,
        is_cubing=data_args.is_cubing,
        )

    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

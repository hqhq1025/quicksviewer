import torch
from transformers import AutoModel, AutoTokenizer
import os,sys, re
import json
import numpy as np
import string
import math
import tqdm
import glob
import itertools
import multiprocessing
from PIL import Image
from argparse import ArgumentParser
from decord import VideoReader, cpu
from pathlib import Path
from typing import Dict, List
import argparse
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

from quicksviewer.utils.mm_utils import get_model_name_from_path
from quicksviewer.utils.utils import disable_torch_init
from quicksviewer.model.builder import load_pretrained_model
from quicksviewer.utils.data_util import opencv_extract_frames_fps
from quicksviewer.eval.utils import extract_characters_regex

class MyDataset(Dataset):
    def __init__(self, json_path, video_dir, samples_range, video_nframes, video_fps,):
        self.video_dir = video_dir
        self.video_nframes = video_nframes
        self.video_fps = video_fps

        self.data_list = {
            "Action Sequence": ("action_sequence.json", f"{video_dir}/star/Charades_v1_480/", "video", True), # has start & end
            "Action Prediction": ("action_prediction.json", f"{video_dir}/star/Charades_v1_480/", "video", True), # has start & end
            "Action Antonym": ("action_antonym.json", f"{video_dir}/ssv2_video/", "video", False),
            "Fine-grained Action": ("fine_grained_action.json", f"{video_dir}/Moments_in_Time_Raw/videos/", "video", False),
            "Unexpected Action": ("unexpected_action.json", f"{video_dir}/FunQA_test/test/", "video", False),
            "Object Existence": ("object_existence.json", f"{video_dir}/clevrer/video_validation/", "video", False),
            "Object Interaction": ("object_interaction.json", f"{video_dir}/star/Charades_v1_480/", "video", True), # has start & end
            "Object Shuffle": ("object_shuffle.json", f"{video_dir}/perception/videos/", "video", False),
            "Moving Direction": ("moving_direction.json", f"{video_dir}/clevrer/video_validation/", "video", False),
            "Action Localization": ("action_localization.json", f"{video_dir}/sta/sta_video/", "video", True),  # has start & end
            "Scene Transition": ("scene_transition.json", f"{video_dir}/scene_qa/video/", "video", False),
            "Action Count": ("action_count.json", f"{video_dir}/perception/videos/", "video", False),
            "Moving Count": ("moving_count.json", f"{video_dir}/clevrer/video_validation/", "video", False),
            "Moving Attribute": ("moving_attribute.json", f"{video_dir}/clevrer/video_validation/", "video", False),
            "State Change": ("state_change.json", f"{video_dir}/perception/videos/", "video", False),
            "Fine-grained Pose": ("fine_grained_pose.json", f"{video_dir}/nturgbd/", "video", False),
            "Character Order": ("character_order.json", f"{video_dir}/perception/videos/", "video", False),
            "Egocentric Navigation": ("egocentric_navigation.json", f"{video_dir}/vlnqa/", "video", False),
            "Episodic Reasoning": ("episodic_reasoning.json", f"{video_dir}/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
            "Counterfactual Inference": ("counterfactual_inference.json", f"{video_dir}/clevrer/video_validation/", "video", False),
        }

        dataset = []
        tot = 0
        for k, v in self.data_list.items():
            with open(os.path.join(json_path, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                # get question, answer
                question, answer = self.qa_template(data)
                video_path = os.path.join(v[1], data['video'])
                dataset.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
                    'question_id': tot,
                    'question': question,
                    'answer': answer,
                    'video_file': video_path,
                    'data': data,
                })
                tot += 1
        self.samples_range = list(map(lambda x: int(x) if x!='-1' else len(dataset), samples_range.split('_')))
        self.dataset = dataset[self.samples_range[0] : self.samples_range[1]]
        print(f"Loading {len(self.dataset)} of {len(dataset)} for this evaluation ..")


    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        # answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        answer = f"({chr(ord('A') + answer_idx)})"
        return question, answer
    
    def read_frame(self, frames_path, num_frames, bound, fps=3):
        timestamps = []
        max_frame = len(os.listdir(frames_path))
        images_group = list()
        # get indices
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(1, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_frames
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_frames)
        ])
        
        for frame_index in frame_indices:
            img_path = os.path.join(frames_path, f"{frame_index:05d}.jpg")
            if not os.path.exists(img_path):
                print(f"Frame not found for {img_path} So ignoring !!!")
                continue
            img = Image.open(img_path)
            images_group.append(np.array(img))
            timestamps.append(round(frame_index/fps, 2))
        video = np.stack(images_group)
        return video, timestamps

    def __getitem__(self, index):
        item = self.dataset[index]
        # Load video
        try:
            if item['data_type'] == 'frame':
                video, timestamps = self.read_frame(item['video_file'], self.video_nframes, (item['data']['start'], item['data']['end']) if item['bound'] else None, self.video_fps)
            else:
                video, timestamps = opencv_extract_frames_fps(item['video_file'], self.video_nframes, self.video_fps, to_pilimg=False)
                video = np.stack(video, axis=0)
        except BaseException as e:
            video = np.zeros([2, 384,384,3], dtype=np.uint8)
            timestamps = [0.0, 1.0]
            print(f"Using dummy video for {item['video_file']}")
        ex = {
            'video': video,
            'timestamps': timestamps,
            'question': item['question'],
            'question_id': item['question_id'],
            'answer': item['answer'],
            'task_type': item['task_type']
        }
        return ex
    
    def __len__(self):
        return len(self.dataset)
    
    


def run(args, dataloader, save_f=None) -> List:
    """
    Params:
        @vqas: [{'video_path', 'question_id', 'question', 'answer[optional]', ...}, ...]
    """
    # Load Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base,
        model_name, args.version, args.load_8bit, args.load_4bit,
        vpm_device=args.vpm_device, llm_device=args.llm_device,
        overwrite_config={'_attn_implementation':"flash_attention_2"}
    )

    # Evaluate
    results = []
    for batch in tqdm.tqdm(dataloader, desc=f"Evaluating on MVBench"):
        for ex in batch:
            video, timestamps, question = ex['video'], ex['timestamps'], ex['question']
            modality = 'video'
            if len(video) <= 1:
                modality = 'image'
                video = Image.fromarray(video[0]).convert('RGB')
            question = question + '\n' + args.prompt if args.prompt else question
            msgs = [{'from':'human', 'value': [video] + [question], 'timestamps': timestamps},
                    {'from': 'gpt', 'value':''}]

            with torch.inference_mode():
                try:
                    outputs = model.chat(
                        image=None,
                        msgs=msgs, # [{'from':'human'|'gpt','value':str|tuple, 'timestamps':list}]
                        modalities=modality, # ['image', 'video', ..]
                        tokenizer=tokenizer,
                        image_processor=image_processor,
                        dtype=torch.float16,
                        llm_device=torch.device(f'cuda:{args.llm_device}') if args.vpm_device!=args.llm_device else None
                    )
                except BaseException as e:
                    print(f"Error in model.chat: {e}, skipping the output ..")
                    outputs = ""
                # print(outputs)
            # print(f"Question: {question}\t Answer: {gt}\t Pred: {res}")
            results.append({
                'pred': outputs,
                "question_id": ex['question_id'],
                'question':ex['question'],
                'answer': ex.get('answer', None),
                'task_type': ex.get('task_type', None),
                })
    if save_f is not None:
        with open(save_f, 'w') as f:
            f.write(json.dumps(results))
    return results


    
def calculate_metrics(
        res_dict_list: List[Dict],
        save_f: str=None) -> Dict:
    """
    Params:
      @res_dict_list: [{'question', 'question_id', 'pred', 'truth'}]
    """
    # tot_correct, tot = 0, 0
    tot_correct, tot = defaultdict(int), defaultdict(int) # per task-type
    for ex in res_dict_list:
        task_type = ex['task_type']
        truth = ex['answer']
        pred = ex['pred'][0] if isinstance(ex['pred'], list) else ex['pred']
        try:
            pred = extract_characters_regex(pred)
        except:
            import ipdb; ipdb.set_trace()
        if pred == "":
            continue
        truth = truth.strip()
        if truth.startswith('(') and truth.endswith(')'):
            truth = truth[1:-1]
        # tot += 1
        tot[task_type] += 1

        if pred == truth:
            # tot_correct += 1
            tot_correct[task_type] += 1

    metrics = {}
    sum_tc, sum_t = 0., 0.
    for task_type in tot:
        acc = tot_correct[task_type] / tot[task_type]
        metrics[task_type] = acc
        sum_tc, sum_t = sum_tc+tot_correct[task_type], sum_t+tot[task_type]
    metrics['avg_acc'] = sum_tc / sum_t
    print(str(metrics))
    if save_f is not None:
        with open(save_f, 'w') as f:
            f.write(json.dumps(metrics))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/quicksviewer-s3/checkpoint-15408")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--version", type=str, default='qwen2')
    parser.add_argument("--data_path", type=str, default='MVBench/MVBench/json/')
    parser.add_argument("--video_dir", type=str, default='MVBench/MVBench/video')
    parser.add_argument("--samples_range", type=str, default='0_-1')
    parser.add_argument("--video_nframes", type=int, default=420)
    parser.add_argument("--video_fps", type=int, default=1)
    parser.add_argument("--vpm-device", type=int, default=0)
    parser.add_argument("--llm-device", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2) # set to 0.5 for video
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--prompt", type=str, default='')
    parser.add_argument("--task", type=str, default='mvbench')
    parser.add_argument("--save_dir", type=str, default='./output/results/')
    parser.add_argument("--calc_metrics", action='store_true')
    args = parser.parse_args()


    dataset = MyDataset(
        json_path=args.data_path,
        video_dir=args.video_dir,
        samples_range=args.samples_range,
        video_nframes=args.video_nframes,
        video_fps=args.video_fps,
    )
    dataloader = DataLoader(dataset, batch_size=8, num_workers=4, collate_fn=lambda x:x)

    os.makedirs(args.save_dir, exist_ok=True)
    modelname = get_model_name_from_path(args.model_path)
    save_result = os.path.join(args.save_dir, f'{modelname}_{args.task}_result_fps={args.video_fps}_nframs={args.video_nframes}.json.{args.samples_range}')
    save_metrics = os.path.join(args.save_dir, f'{modelname}_{args.task}_metrics_fps={args.video_fps}_nframs={args.video_nframes}.json')

    if not os.path.exists(save_result):
        results = run(args, dataloader)
        # save
        with open(save_result, 'w') as f:
            f.write(json.dumps(results))
        
    # Calculate metrics
    if args.calc_metrics: # Find all results splits
        results = []
        for f in glob.glob(".".join(save_result.split('.')[:-1]) + '*'):
            results.extend(json.load(open(f)))
        calculate_metrics(results, save_metrics)

    

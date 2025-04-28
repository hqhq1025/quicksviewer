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

from quicksviewer.utils.mm_utils import get_model_name_from_path
from quicksviewer.utils.utils import disable_torch_init
from quicksviewer.model.builder import load_pretrained_model
from quicksviewer.utils.data_util import opencv_extract_frames_fps
from quicksviewer.eval.utils import extract_characters_regex

class MyDataset(Dataset):
    def __init__(self, parquet_path, video_dir, samples_range, video_nframes, video_fps,):
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        self.samples_range = list(map(lambda x: int(x) if x!='-1' else len(df), samples_range.split('_')))
        self.dataset = df[self.samples_range[0] : self.samples_range[1]]
        self.video_dir = video_dir
        self.video_nframes = video_nframes
        self.video_fps = video_fps

    def __getitem__(self, index):
        item = self.dataset.iloc[index]
        # Get question
        question = f"Carefully read the following question and select the letter corresponding to the correct answer.Highlight the applicable choices without giving explanations.\n{item.question}\nOptions:"
        for ai in range(len(item.options)):
            question += f"\n{item.options[ai]}"
        # Load video
        try:
            video, timestamps = opencv_extract_frames_fps(self.video_dir + item.videoID + '.mp4', self.video_nframes, self.video_fps, to_pilimg=False)
            video = np.stack(video, axis=0)
        except BaseException as e:
            video = np.zeros([self.video_nframes, 384,384,3], dtype=np.uint8)
            timestamps = [0.0]
            print(f"Using dummy video for {item.videoID}")
        ex = {
            'video': video,
            'timestamps': timestamps,
            'question': question,
            'question_id': item.question_id,
            'answer': item.answer,
            'video_id': item.videoID,
            'duration': item.duration
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
    for batch in tqdm.tqdm(dataloader, desc=f"Evaluating on VideoMME"):
        for ex in batch:
            video, timestamps, question = ex['video'], ex['timestamps'], ex['question']
            question = question + '\n' + args.prompt if args.prompt else question
            msgs = [{'from':'human', 'value': [video] + [question], 'timestamps': timestamps},
                    {'from': 'gpt', 'value':''}]

            with torch.inference_mode():
                try:
                    outputs = model.chat(
                        image=None,
                        msgs=msgs, # [{'from':'human'|'gpt','value':str|tuple, 'timestamps':list}]
                        modalities='video', # ['image', 'video', ..]
                        tokenizer=tokenizer,
                        image_processor=image_processor,
                        dtype=torch.float16,
                        llm_device=torch.device(f'cuda:{args.llm_device}') if args.vpm_device!=args.llm_device else None
                    )
                except BaseException as e:
                    print(f"Error in model.chat: {e}, skipping the output ..")
                    outputs = ""
            results.append({
                'pred': outputs,
                "question_id": ex['question_id'],
                'question':ex['question'],
                'answer': ex.get('answer', None),
                'video_id': ex.get('video_id', None),
                'duration': ex.get('duration', None),
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
    tot, tot_correct = 0, 0
    for ex in res_dict_list:
        truth = ex['answer']
        pred = ex['pred']
        if isinstance(pred, list):
            pred = pred[0]
        pred = extract_characters_regex(pred)
        # change for subsequent calculation
        ex['pred'] = pred.lower()
        ex['answer'] = truth.lower()
        tot += 1
        if truth.lower() == pred.lower():
            tot_correct += 1

    acc = tot_correct / tot
    metrics = {'Tot': tot, 'Acc': acc}
    for dur in ['short', 'medium', 'long']:
        dur_acc = np.mean([i['pred']==i['answer'] for i in res_dict_list if i['duration'] == dur])
        metrics[f'{dur}_acc'] = dur_acc
    print(str(metrics))

    if save_f is not None:
        with open(save_f, 'w') as f:
            f.write(json.dumps(metrics))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/quicksviewer-s3/checkpoint-15408")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--version", type=str, default='qwen2')
    parser.add_argument("--data_path", type=str, default='Video-MME/videomme/test-00000-of-00001.parquet')
    parser.add_argument("--video_dir", type=str, default='Video-MME/video_mme_video/')
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
    parser.add_argument("--task", type=str, default='videomme')
    parser.add_argument("--save_dir", type=str, default='./output/results/')
    parser.add_argument("--calc_metrics", action='store_true')
    args = parser.parse_args()


    dataset = MyDataset(
        parquet_path=args.data_path,
        video_dir=args.video_dir,
        samples_range=args.samples_range,
        video_nframes=args.video_nframes,
        video_fps=args.video_fps,
    )
    dataloader = DataLoader(dataset, batch_size=4, num_workers=4, collate_fn=lambda x:x)

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

    

import os, sys
import json
from datasets import load_dataset
import multiprocessing
from multiprocessing.pool import ThreadPool
import itertools
import tqdm
from glob import glob
from pathlib import Path
import pandas as pd
import re
import argparse

from quicksviewer.preprocess.utils import split_list
from quicksviewer.preprocess.template import build_with_options
from quicksviewer.utils.mm_utils import is_video
from quicksviewer.preprocess.change_datasets import check_valid_video


class LongVilaSFT():
    def __init__(self, ann_path="longvila_sft_dataset/data/train-00000-of-00001.parquet",
                 video_dir="longvila_sft_dataset/videos720/",
                 output_dir='vid_train_datasets/video_data/',
                 num_workers=1):
        self.ann_path = ann_path
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.prefix = 'LongVilaSFT'
        self.save_results = True
        # Load video paths
        self.vid2path = {}
        for root, folder, files in tqdm.tqdm(list(os.walk(self.video_dir)), desc="Loading video paths"):
            for file in files:
                if is_video(file):
                    with open(os.path.join(root, Path(file).with_suffix('.json'))) as f: meta=json.load(f)
                    vname = meta.get('yt_meta_dict', {}).get('info', {}).get('id', '')
                    if vname != '':
                        # self.vname2path[vname + Path(file).suffix] = os.path.join(root, file)
                        self.vid2path[vname] = os.path.join(root, file)

    def build(self):
        """ Build dataset with the unified format.
        """
        # convert
        data = load_dataset('parquet',
                             data_files={'train': f'{self.ann_path}'})
        data_pd = data['train'].to_pandas()
        if self.num_workers <=1:
            result = self._build_samples(data_pd)
        else:
            trunks = split_list(data_pd, self.num_workers)
            pool = multiprocessing.Pool(processes=self.num_workers)
            # pool = ThreadPool(processes=self.num_workers)
            result = [pool.apply_async(self._build_samples, args=[trunks[i]]) for i in range(self.num_workers)]
            pool.close(); pool.join()
            result = list(itertools.chain(*[rt.get() for rt in result]))
        # Save
        if self.save_results:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, f'{self.prefix}.jsonl'), 'w') as f:
                for ex in result:
                    f.write(json.dumps(ex)+'\n')
        print(f"Completed to process {self.ann_path}")
    

    def _build_samples(self, dataframe):
        """ ori_sample in longvilasft: {'id':vid, 'video':vid.mp4, 'conversations':{'from':'gpt'|'human', 'value':'xxx \n<video>'}}
        """
        result = []
        for i, row in tqdm.tqdm(enumerate(dataframe.iterrows()), desc="processing LongVilaSFT"):
            vid, vname, conversations = row[1]['id'], row[1]['video'], row[1]['conversations'].tolist()
            vpath = self.vid2path.get(vid, '')
            if vpath=='':
                print(f"Not found for video: {vid}")
                continue
            if not check_valid_video(vpath):
                print(f"The video is invalid: {vpath}")
                continue

            # Build QA sample
            for conv in conversations:
                if conv['from'] == 'human' and '<video>' in conv['value']:
                    conv['value'] = re.sub(r'\s*<video>', '',conv['value'], count=1)

            result.append({
                'uid': f'{self.prefix}_{vid}_{i}',
                'video': vpath,
                'conversations': conversations,
                'data_type': 'conversations',
            })
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_path', default="longvila_sft_dataset/data/train-00000-of-00001.parquet")
    parser.add_argument('--video_dir', default="longvila_sft_dataset/videos720/")
    parser.add_argument('--output_dir', default='vid_train_datasets/video_data/')
    parser.add_argument('--num_workers', type=int, default=120)
    args = parser.parse_args()

    dataset = LongVilaSFT(args.ann_path, args.video_dir, args.output_dir, args.num_workers)
    dataset.build()
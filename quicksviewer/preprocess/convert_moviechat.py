import os, sys
import json
import random
from datasets import load_dataset
from multiprocessing.pool import ThreadPool
import itertools
import tqdm
from glob import glob
from pathlib import Path

from quicksviewer.preprocess.utils import split_list
from quicksviewer.preprocess.template import DETAIL_DESC_VIDEO_EN


class MovieChat():
    def __init__(self, ann_dir="MovieChat-1K_train/jsons/",
                 video_dir = 'MovieChat-1K_train/raw_videos/',
                 output_dir='vid_train_datasets/video_data/',
                 num_workers=1):
        self.ann_dir = ann_dir
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.prefix = 'MovieChat'
        self.save_results = True

    def build(self):
        """ Build dataset with the unified format.
        """
        # convert
        jsons = list(glob(os.path.join(self.ann_dir, f'*.json')))
        if self.num_workers <=1:
            result = self._build_samples(jsons)
        else:
            trunks = split_list(jsons, self.num_workers)
            # pool = multiprocessing.Pool(processes=self.workers)
            pool = ThreadPool(processes=self.num_workers)
            result = [pool.apply_async(self._build_samples, args=[trunks[i]]) for i in range(self.num_workers)]
            pool.close(); pool.join()
            result = list(itertools.chain(*[rt.get() for rt in result]))
        # Save
        if self.save_results:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, f'{self.prefix}.jsonl'), 'w') as f:
                for ex in result:
                    f.write(json.dumps(ex)+'\n')
        print(f"Completed to process {self.ann_dir}")
    

    def _build_samples(self, jsons):
        """ Useful field in MovieChat: caption, global[question,answer].
          Build samples with 2 types:
            caption: {human: i, gpt: caption}
            QAs: {human: q, gpt: a}
        """
        result = []
        for i, jsf in tqdm.tqdm(enumerate(jsons), desc="processing MovieChat"):
            fname = Path(jsf).stem
            try:
                with open(jsf) as f:
                    ex = json.load(f)
            except BaseException as e:
                print(f"Faild to load json file {jsf}, exception: {e}")
                continue
            # Extract & Save video
            vpath = os.path.join(self.video_dir, fname + f'.mp4')
            if not os.path.exists(vpath):
                print(f"Not found for video path: {vpath}")

            # Build caption sample
            result.append({
                'uid': f'{self.prefix}_{fname}_caption_{i}',
                'video': vpath,
                'conversations': [{'from':'human', 'value':random.choice(DETAIL_DESC_VIDEO_EN)}, {'from':'gpt','value':ex['caption']}],
                'data_type': 'conversations',
            })
            # Build QA sample
            for qa in ex['global']:
                result.append({
                    'uid': f'{self.prefix}_{fname}_qa_{i}',
                    'video': vpath,
                    'conversations': [{'from':'human', 'value':qa['question']}, {'from':'gpt','value':qa['answer']}],
                    'data_type': 'conversations',
                })
        return result


if __name__ == '__main__':
    dataset = MovieChat(num_workers=20)
    dataset.build()
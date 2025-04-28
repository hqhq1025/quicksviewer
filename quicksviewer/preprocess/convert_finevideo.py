import multiprocessing.pool
import os, sys
import json
from datasets import load_dataset
import itertools
import tqdm
from glob import glob
from pathlib import Path
import multiprocessing
from multiprocessing.pool import ThreadPool
from multiprocessing import Lock
import re
import argparse

from quicksviewer.preprocess.utils import split_list
from quicksviewer.utils.data_util import opencv_extract_frames_fps
from quicksviewer.preprocess.convert_2tsv import save_2tsv_dataset




class FineVideo():
    def __init__(self,
                ann_path="finevideo/finevideo_filtered.jsonl",
                video_dir="finevideo/videos",
                output_dir='vid_train_datasets/tsv_data/FineVideo',
                num_workers=40,
                samples_range=""):
        self.ann_path = ann_path
        self.video_dir = video_dir
        self.samples_range = [int(x) if x!='-1' else None for x in samples_range.split('_')] if samples_range else [0,None] # to process how much samples
        self.output_dir = os.path.join(output_dir, samples_range)
        self.num_workers = num_workers
        self.prefix = 'FineVideo'
        self.save_results = True # save both TSVs and Chat.jsonl

    def build(self):
        """ Build dataset with the unified format.
        """
        # convert
        with open(self.ann_path) as f:
            data = [json.loads(line) for line in f]
            data = data[self.samples_range[0]: self.samples_range[1]]
        if self.num_workers <=1:
            result = self._build_samples(data)
        else:
            trunks = split_list(data, self.num_workers)
            lock = multiprocessing.Manager().Lock()
            pool = multiprocessing.Pool(processes=self.num_workers)
            # pool = ThreadPool(processes=self.num_workers)
            result = [pool.apply_async(self._build_samples, args=[trunks[i], lock]) for i in range(self.num_workers)]
            pool.close(); pool.join()
            result = list(itertools.chain(*[rt.get() for rt in result]))
        # Save
        if self.save_results:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, f'{self.prefix}_caption_storyline.jsonl'), 'w') as f1, \
                open(os.path.join(self.output_dir, f'{self.prefix}_qa_climax.jsonl'), 'w') as f2:
                for ex in result:
                    if 'images' in ex:
                        ex.pop('images') # pop video_bytes
                    if ex['category'] in ['Caption', 'Storyline']:
                        f1.write(json.dumps(ex)+'\n')
                    elif ex['category'] in ['QA', 'Climax']:
                        f2.write(json.dumps(ex)+'\n')
        print(f"Completed to process {self.ann_path}")

    def _build_samples(self, samples, lock=None):
        """ Convert to CSV.
        """
        save_interval, saved = 100, 0
        result = []
        for i, ex in tqdm.tqdm(enumerate(samples), desc="processing FineVideo"):
            vpath = os.path.join(self.video_dir,Path(ex['video']).name)
            if not os.path.exists(vpath):
                print(f"Warning: video {vpath} not found!!")
                continue

            # load video
            attempts, successful = 10, False
            while attempts > 0:
                try:
                    res_frames, timestamps, video_bytes = opencv_extract_frames_fps(vpath,fps=1, to_base64=True, to_pilimg=False)
                    successful = True
                    break
                except BaseException as e:
                    print(f"Failed to load video {vpath}, retrying ...")
                    attempts -= 1
            if not successful:
                continue

            # save tsv
            ex['images'] = video_bytes
            ex['timestamps'] = timestamps
            if 'qAndA' in ex['uid'] or re.match(r'^Please locate the timestamp.*',ex['conversations'][0]['value']):
                ex['category'] = 'QA' if 'qAndA' in ex['uid'] else 'Climax'
            else:
                ex['category'] = 'Storyline' if re.match(r'^Please provide a general outline of the storyline.*', ex['conversations'][0]['value']) else 'Caption'
            result.append(ex)
            
            # Save
            if len(result[saved:]) % save_interval == 0 or i==len(samples)-1:
                if lock is not None:
                    lock.acquire()
                if len(result[saved:]) > 0:
                    save_2tsv_dataset(result[saved:], os.path.join(self.output_dir))
                if lock is not None:
                    lock.release()
                saved = len(result)
        return result
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_path', default="finevideo/finevideo_filtered.jsonl")
    parser.add_argument('--video_dir', default="finevideo/videos")
    parser.add_argument('--output_dir', default='vid_train_datasets/tsv_data/')
    parser.add_argument('--tsv_output_dir', default='vid_train_datasets/tsv_data/FineVideo')
    parser.add_argument('--num_workers', type=int, default=40)
    parser.add_argument('--samples_range', type=str, default="0_348814")
    args = parser.parse_args()

    dataset = FineVideo(args.ann_path, args.video_dir, args.tsv_output_dir, args.num_workers, args.samples_range)
    dataset.build()

import os,sys
import json
import tqdm
import multiprocessing
from multiprocessing.pool import ThreadPool
from multiprocessing import Lock
import itertools
from pathlib import Path
import re
from collections import defaultdict
import argparse

from quicksviewer.preprocess.utils import split_list
from quicksviewer.utils.data_util import opencv_extract_frames_fps
from quicksviewer.preprocess.convert_2tsv import save_2tsv_dataset
from quicksviewer.utils.mm_utils import is_video



class DatasetBase:
    def __init__(self, num_workers):
        self.num_workers = num_workers
    
    def _build_samples(self, samples, lock):
        raise NotImplementedError

    def build(self):
        """ Build dataset with the unified format.
        """
        # convert
        with open(self.ann_path) as f:
            data = json.load(f)
            data = [(vname, data[vname]) for vname in data] # for segment simply
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
            with open(os.path.join(self.output_dir, f'{self.prefix}.jsonl'), 'w') as f:
                for ex in result:
                    f.write(json.dumps(ex)+'\n')
        print(f"Completed to process {self.ann_path}")


class ANetCaptions(DatasetBase):
    def __init__(self,
                ann_path="sharegpt4video_40k.jsonl",
                video_dir="sharegpt4video/videos/",
                output_dir='vid_train_datasets/tsv_data/ShareGPT4Video',
                num_workers=40):
        super().__init__(num_workers)
        self.ann_path = ann_path
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.prefix = 'ANetCaptions'
        self.save_results = False

    def _build_samples(self, samples, lock=None):
        """ Convert to CSV.
        """
        result = []
        save_interval = 100
        for i, (vname, ex) in tqdm.tqdm(enumerate(samples), desc="processing ANetCaptions"):
            vpath = os.path.join(self.video_dir, vname+'.mp4')
            if not os.path.exists(vpath):
                print(f"Warning: video {vpath} not found!!")
                continue

            # merge caption
            caption = " ".join(ex['sentences'])
            # save tsv
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
            new_ex = {
                'uid': f'{self.prefix}_{vname}',
                'video': vpath,
                'caption': caption,
                'data_type': 'caption',
            }
            new_ex['images'] = video_bytes
            new_ex['timestamps'] = timestamps
            result.append(new_ex)
            # Save
            # if i % save_interval == 0 or i==len(samples)-1:
            if len(result) % save_interval == 0 or i==len(samples)-1:
                if lock is not None:
                    lock.acquire()
                if len(result) > 0:
                    save_2tsv_dataset(result, self.output_dir)
                if lock is not None:
                    lock.release()
                result = []
        return []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_path', default="ActivityNet/captions/train.json")
    parser.add_argument('--video_dir', default="activitynet/train/")
    parser.add_argument('--output_dir', default='vid_train_datasets/tsv_data/ANetCaptions')
    parser.add_argument('--num_workers', type=int, default=120)
    args = parser.parse_args()

    dataset = ANetCaptions(args.ann_path, args.video_dir, args.output_dir, args.num_workers)
    dataset.build()
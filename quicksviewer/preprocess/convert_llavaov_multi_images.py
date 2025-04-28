""" The data source is M4-Instruct (used in LLaVA-OneVision): https://huggingface.co/datasets/lmms-lab/M4-Instruct-Data
"""
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
from quicksviewer.preprocess.change_datasets import check_valid_image



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
                ann_path="m4_instruct_annotations.json",
                image_dir="M4-Instruct-Data/",
                output_dir='img_train_datasets/',
                num_workers=40):
        super().__init__(num_workers)
        self.ann_path = ann_path
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.prefix = 'LLaVAOneVision-MultiImages'
        self.save_results = True

    def _build_samples(self, samples, lock=None):
        """ Convert to CSV.
        """
        result = []
        i = 0
        for ex in tqdm.tqdm(samples, desc="processing LLaVAOneVision-MultiImages"):
            impaths = [os.path.join(self.image_dir, img) for img in ex['image']]
            # Filter  (NOTE that: time costuming!)
            filter_result = [check_valid_image(img) for img in impaths]
            if not all(filter_result):
                print(f"Warning: skipping invalid images {[impaths[_] for _ in range(len(filter_result)) if not filter_result[_]]}")
                continue
            # convert
            ex['image'] = impaths
            ex['data_type'] = 'conversations'
            result.append(ex)
            i += 1
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_path', default="M4-Instruct-Data/m4_instruct_annotations.json")
    parser.add_argument('--image_dir', default="M4-Instruct-Data/")
    parser.add_argument('--output_dir', default='img_train_datasets/')
    parser.add_argument('--num_workers', type=int, default=80)
    args = parser.parse_args()

    dataset = ANetCaptions(args.ann_path, args.image_dir, args.output_dir, args.num_workers)
    dataset.build()
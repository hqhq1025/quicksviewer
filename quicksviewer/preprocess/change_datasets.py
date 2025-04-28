import os, sys
import json
import re
from datasets import load_dataset
from multiprocessing.pool import ThreadPool
import multiprocessing
import itertools
import tqdm
from glob import glob
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
import argparse
from datetime import datetime

from quicksviewer.preprocess.utils import split_list
from quicksviewer.utils.mm_utils import is_video, is_image
from quicksviewer.utils.data_util import opencv_extract_frames_fps


def check_valid_video(vpath, valid_frames_range=[2,-1],):
    valid = is_video(vpath) and os.path.exists(vpath)
    if valid:
        try:
            video_frames, timestamps = opencv_extract_frames_fps(vpath)
            # if len(video_frames) < 2:
            if len(video_frames) < valid_frames_range[0]:
                valid = False
            if valid_frames_range[1] > 0 and len(video_frames) > valid_frames_range[1]:
                valid = False
        except BaseException as e:
            valid = False
    return valid


def check_valid_image(impath):
    valid = is_image(impath) and os.path.exists(impath)
    if valid:
        try:
            pil_img = Image.open(impath)
            if np.array(pil_img).dtype != np.uint8:
                valid = False
        except BaseException as e:
            valid = False
    return valid


def check_valid_roles(ex):
    valid_roles = ["human", "gpt"]
    valid = True
    for conv in ex.get('conversations', []):
        try:
            if conv['from'] not in valid_roles or conv['value']=="":
                valid = False
                break
        except BaseException:
            valid = False
    return valid

def check_valid_len_img(ex, valid_len_range=[1,16384], scale_factor=1.5, ntokens_per_img=64, npatches_per_img=5):
    valid = True
    cap_len = 0
    if ex['data_type'] == 'conversations':
        cap_len = sum([len(conv['value'].split())*scale_factor for conv in ex['conversations']])
        images = ex['image'] if 'image' in ex else []
    elif ex['data_type'] == 'caption':
        cap_len = len(ex['caption'].split())*scale_factor
        images = ex['image'] if 'image' in ex else []
    elif ex['data_type'] == 'interleave':
        cap_len = sum([len(cap.split())*scale_factor for cap in ex['caption'] if cap is not None])
        images = [img for img in ex['image'] if img is not None]
    images = [images] if not isinstance(images, list) else images
    im_len = sum(npatches_per_img*ntokens_per_img for _ in range(len(images)))
    tot_len = cap_len + im_len
    # check cap
    if cap_len < 1:
        valid = False
    # check total
    if tot_len < valid_len_range[0] or tot_len > valid_len_range[1]:
        valid = False
    return valid

def check_valid_modality(ex, valid_fields=['image', 'video']):
    return any([ex.get(field) is not None for field in valid_fields])

def check_valid_placeholder(ex, placeholder='<image>'):
    valid = True
    if ex['data_type'] == 'conversations':
        valid = any([placeholder in conv['value'] for conv in ex['conversations']]) # placeholder
    return valid

class DatasetBase():
    def __init__(self, num_workers, save_results=True, resave_name='new'):
        self.num_workers = num_workers
        self.save_results = save_results
        self.resave_name = resave_name

    def process_samples(self, samples, data_path, pro_func, *args, **kwargs):
        if self.num_workers <=1:
            result = pro_func(samples, *args, **kwargs)
        else:
            trunks = split_list(samples, self.num_workers)
            pool = multiprocessing.Pool(processes=self.num_workers)
            # pool = ThreadPool(processes=self.num_workers)
            result = [pool.apply_async(pro_func, args=[trunks[i], *args], kwds=kwargs) for i in range(self.num_workers)]
            pool.close(); pool.join()
            result = list(itertools.chain(*[rt.get() for rt in result]))
        # Save
        if self.save_results:
            with open(Path(data_path).with_stem(Path(data_path).stem+f'_{self.resave_name}'), 'w') as f:
                for ex in result:
                    f.write(json.dumps(ex)+'\n')


class Dataset(DatasetBase):
    def __init__(self, root_dir="/data/vid_train_datasets/video_data/",
                 keywords = ['finevideo.jsonl'], skip_keywords = ['_filtered'],
                 resave_name='filtered',
                 num_workers=1,):
        super().__init__(num_workers, save_results=True, resave_name=resave_name)
        self.root_dir = root_dir
        self.keywords = keywords
        self.skip_keywords = skip_keywords
        # self.resave_name = resave_name
        # self.num_workers = num_workers
        # self.save_results = True

        # load all files to be processed
        self.files = []
        for root, folders, files in os.walk(root_dir):
            for fname in files:
                if any(map(lambda x: x in fname, keywords)) \
                    and all(map(lambda x: x not in fname, skip_keywords)):
                    self.files.append(os.path.join(root, fname))


    def filter(self, fields=['video']):
        """ filter out invalid samples and re-save.
        """
        # load all examples
        for data_path in self.files:
            samples = []
            print(f"Start to process {data_path}")
            with open(data_path) as f:
                for line in f:
                    samples.append(json.loads(line))
            self.process_samples(samples, data_path, self._filter_samples, fields=fields)
            print(f"Completed to process {data_path}")
    

    # def _filter_samples(self, samples, fields, check_func):
    def _filter_samples(self, samples, fields):
        result = []
        for ex in tqdm.tqdm(samples, desc="filtering"):
            is_valid = True
            for field in fields:
                if field in ex:
                    if is_image(ex[field]):
                        is_valid = check_valid_image(ex[field])
                    elif is_video(ex[field]):
                        is_valid = check_valid_video(ex[field])
            is_valid = is_valid and check_valid_roles(ex)
            # is_valid = is_valid and check_valid_modality(ex)
            # is_valid = is_valid and check_valid_len_img(ex)
            # is_valid = is_valid and check_valid_placeholder(ex)
            if is_valid:
                result.append(ex)
            else:
                # print(f"Invalid media for {[ex.get(f, None) for f in fields]}")
                print(f"Invalid sample: {ex}")
        return result
    

    def change_path(self, fields=['video'], src_ptr=r'', tgt_str=''):
        # load all examples
        for data_path in self.files:
            print(f"Start to process {data_path}")
            with open(data_path) as f:
                samples = [json.loads(line) for line in f]
                self.process_samples(samples, data_path, self._change_path_samples, fields=fields, src_ptr=src_ptr, tgt_str=tgt_str)
            print(f"Completed to process {data_path}")

    
    def _change_path_samples(self, samples, fields, src_ptr, tgt_str):
        result = []
        for ex in samples:
            for field in fields:
                if field in ex:
                    if isinstance(ex[field], str):
                        ex[field] = re.sub(src_ptr, tgt_str, ex[field])
                    elif isinstance(ex[field], list):
                        ex[field] = [re.sub(src_ptr, tgt_str, s) if isinstance(s, str) else s for s in ex[field]]
            result.append(ex)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="")
    parser.add_argument('--keywords', type=str, nargs='+', default=['.jsonl'])
    parser.add_argument('--skip_keywords', type=str, nargs='+', default=['_newpath'])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--func', type=str, default="change_path")
    parser.add_argument('--fields', type=str, nargs='+', default=['video', 'image'])
    parser.add_argument('--src_ptr', type=str, default=r'')
    parser.add_argument('--tgt_str', type=str, default='')
    args = parser.parse_args()

    _stime = datetime.now()
    if args.func == 'filter':
        dataset = Dataset(root_dir=args.root_dir, resave_name='filtered',
                          keywords = args.keywords, skip_keywords = args.skip_keywords,
                          num_workers=args.num_workers)
        dataset.filter(args.fields)
    elif args.func == 'change_path':
        dataset = Dataset(root_dir=args.root_dir, resave_name='newpath',
                        keywords = args.keywords, skip_keywords = args.skip_keywords,
                        num_workers=args.num_workers)
        dataset.change_path(args.fields, src_ptr=args.src_ptr, tgt_str=args.tgt_str)

    print(f'Done. Total time consumption: {(datetime.now() - _stime).seconds}s.')
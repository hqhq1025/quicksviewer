import os
from os import path as op
import json
import ipdb.stdout
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import glob
import random
from PIL import Image
from io import BytesIO
from multiprocessing import Pool, cpu_count
import itertools
# import polars as pl
import threading

from template import DESC_EN, IMG_REPLACE, VIDEO_REPLACE
from utils import split_list


def convert_llavapretrain_2pretrain(ann_path, img_dir, save_path):
    """(1) LLaVA-CC3M-Pretrain-595K.jsonl  (2) blip_laion_cc_sbu_558k.json
    """
    print(f"Converting {ann_path} ...")
    results = []
    for ex in tqdm(json.load(open(ann_path))):
        ex.update({
            'image':os.path.join(img_dir, ex['image']),
            'conversations': ex['conversations'],
            'data_type': 'conversations',
        })
        if not op.exists(ex['image']):
            print(f"Not found for {ex['image']}")
            continue
        results.append(ex)
    with open(save_path, 'w') as f:
        for ex in results: f.write(json.dumps(ex)+'\n')

        



if __name__ == '__main__':

    convert_llavapretrain_2pretrain(
        ann_path='LLaVA-CC3M-Pretrain-595K/chat.json',
        img_dir='LLaVA-CC3M-Pretrain-595K/images/',
        save_path='img_train_datasets/LLaVA-CC3M-Pretrain-595K.jsonl'
    )
    convert_llavapretrain_2pretrain(
        ann_path='LLaVA-Pretrain/blip_laion_cc_sbu_558k.json',
        img_dir='LLaVA-Pretrain/images',
        save_path='img_train_datasets/blip_laion_cc_sbu_558k.jsonl'
    )


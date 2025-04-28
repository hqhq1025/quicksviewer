import os, sys
import json
from datasets import load_dataset
import multiprocessing
from multiprocessing.pool import ThreadPool
import itertools
from tqdm import tqdm
from glob import glob
from pathlib import Path
import pandas as pd
import random

from quicksviewer.preprocess.utils import split_list
from quicksviewer.preprocess.template import DESC_EN, IMG_REPLACE
from quicksviewer.utils.mm_utils import is_video
from quicksviewer.preprocess.change_datasets import check_valid_image


class LLaVARecap():
    def __init__(self, data_path="LLaVA-ReCap-558K/data",
                 out_images_dir="LLaVA-ReCap558-images",
                 output_dir='img_train_datasets/',
                 num_workers=1):
        self.data_path = data_path
        self.out_images_dir = out_images_dir
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.prefix = 'LLaVAReCap558K'
        self.save_results = True


    def build(self):
        """ Build dataset with the unified format.
        """
        parquets = glob(os.path.join(self.data_path, f"*.parquet"))
        
        if self.num_workers <=1:
            result = self._build_samples(parquets)
        else:
            trunks = split_list(parquets, self.num_workers)
            # lock = multiprocessing.Manager().Lock()
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
        print(f"Completed to process {self.data_path}")
    

    def _build_samples(self, parquets, lock=None):
        result = []
        for pqf in parquets:
            data = load_dataset('parquet', data_files={'train': pqf})['train']
            for da in tqdm(data, desc="processing LLaVA-Recap-558K"):
                json_data = {}
                json_data["uid"] = da["id"]
                if da["image"] is not None:
                    img_path = os.path.join(self.out_images_dir, f"{da['id']}.jpg")
                    json_data["image"] = img_path
                    os.makedirs(Path(img_path).parent, exist_ok=True)
                    if not os.path.exists(img_path):
                        if lock is not None:
                            lock.acquire()
                        da["image"].convert('RGB').save(img_path)
                        if lock is not None:
                            lock.release()
                if not check_valid_image(img_path):
                    print(f"Invalid image_path: {img_path}")
                    continue
                # Build conversation
                prompt = '<image>\n' + random.choice(DESC_EN).format(image=random.choice(IMG_REPLACE))
                assert '<image>' in da["conversations"][0]['value'], "<image> is not found in conversation"
                da["conversations"][0]['value'] = prompt
                json_data["conversations"] = da["conversations"]
                json_data['data_type'] = 'conversations'
                result.append(json_data)
        return result


if __name__ == '__main__':
    dataset = LLaVARecap(num_workers=26)
    dataset.build()
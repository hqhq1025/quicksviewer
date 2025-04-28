import os, sys
import json
from datasets import load_dataset
from multiprocessing.pool import ThreadPool
import itertools
from tqdm import tqdm
from glob import glob
from pathlib import Path
import pandas as pd

from quicksviewer.preprocess.utils import split_list
from quicksviewer.preprocess.template import build_with_options
from quicksviewer.utils.mm_utils import is_video


class LLaVAOneVision():
    def __init__(self, data_path="LLaVA-OneVision-Data",
                 out_images_dir="LLaVA-OneVision-images",
                 output_dir='img_train_datasets/',
                 num_workers=1):
        self.data_path = data_path
        self.out_images_dir = out_images_dir
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.prefix = 'LLaVAOneVision'
        self.save_results = True


    def build(self):
        """ Build dataset with the unified format.
        """
        data = load_dataset('parquet', data_dir=self.data_path)
        data = data['train']
        
        if self.num_workers <=1:
            result = self._build_samples(data)
        else:
            # trunks = split_list(data, self.num_workers)
            trunks = split_list(range(len(data)), self.num_workers)
            # pool = multiprocessing.Pool(processes=self.workers)
            pool = ThreadPool(processes=self.num_workers)
            # result = [pool.apply_async(self._build_samples, args=[trunks[i]]) for i in range(self.num_workers)]
            result = [pool.apply_async(self._build_samples, args=[data[trunks[i][0]: trunks[i][-1]]]) for i in range(self.num_workers)]
            pool.close(); pool.join()
            result = list(itertools.chain(*[rt.get() for rt in result]))
        # Save
        if self.save_results:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, f'{self.prefix}.jsonl'), 'w') as f:
                for ex in result:
                    f.write(json.dumps(ex)+'\n')
        print(f"Completed to process {self.data_path}")
    

    def _build_samples(self, data):
        """ Convert & Save LLaVA-OneVision-data according to https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data?row=0.
        """
        result = []

        for da in tqdm(data, desc="processing LLaVA-OneVision"):
            json_data = {}
            json_data["uid"] = da["id"]
            if da["image"] is not None:
                img_path = os.path.join(self.out_images_dir, f"{da['id']}.jpg")
                json_data["image"] = img_path
                os.makedirs(Path(img_path).parent, exist_ok=True)
                if not os.path.exists(img_path):
                    da["image"].convert('RGB').save(img_path)
            json_data["conversations"] = da["conversations"]
            json_data['data_type'] = 'conversations'
            result.append(json_data)
        return result


if __name__ == '__main__':
    dataset = LLaVAOneVision(num_workers=1)
    dataset.build()
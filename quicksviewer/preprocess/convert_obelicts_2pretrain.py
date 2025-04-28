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
from multiprocessing.pool import ThreadPool
import itertools
import polars as pl
import threading
import time
import numpy as np

from utils import split_list

def _convert_obelicts_2pretrain(parquets, imdir, savef):
    n_failed = 0
    res = []
    out_stream=open(savef, 'w')
    for i, pqf in enumerate(parquets):
        try:
            dataframe = pl.read_parquet(pqf)
        except Exception as e:
            print(f"Read parquet failed, {e}")
            continue
        for urls, meta, _, texts, image_bytes in tqdm(dataframe.iter_rows(),
                    desc=f"[{i} of {len(parquets)}]", total=len(dataframe)):
            if len(urls)<2 or urls[0] is None: # Filter out begining with img
                continue
            else:
                img_paths = []
                failed = False
                for ii, bys in enumerate(image_bytes):
                    if bys is None:
                        impath=None
                    else:
                        max_trials, succeed = 10, False
                        for i in range(max_trials):
                            try:
                                impath = str(Path(imdir)/'_'.join(Path(urls[ii]).parts[-2:]))
                                img = Image.open(BytesIO(image_bytes[ii]))
                                if not (impath.endswith('jpg') or impath.endswith('png')):
                                    impath += '.jpg' if img.format=='JPEG' else 'png'
                                if not op.exists(impath):
                                    Image.open(BytesIO(image_bytes[ii])).save(impath)
                                succeed = True
                                break
                            except Exception as e:
                                pass
                                # time.sleep(random.random())
                                # print(f"{e}, Retrying saving image {urls[ii]} ...")
                        if not succeed:
                            failed = True
                            n_failed += 1
                            print(f"Fianlly saving image {urls[ii]} failed !")
                            break
                    img_paths.append(impath)
                if failed: continue
                if len(img_paths)>0 and img_paths[-1] is not None: # Cut to remove last img
                    img_paths.pop(-1); texts.pop(-1)
                if len(texts)==len(img_paths) >=2:
                    res.append({
                        'data_type': 'interleave',
                        'caption': texts, # a list of captions
                        'image':  img_paths # a list of paths
                    })
                    out_stream.write(json.dumps(res[-1])+'\n')
                    out_stream.flush()
    print(f"Failed: {n_failed}")
    return res

def convert_obelicts_2pretrain(ann_path, img_dir, save_path, num_workers=0, num_threads=20, nmerge=30):
    print(f"Converting {ann_path} ...")
    os.makedirs(img_dir, exist_ok=True)
    files = [op.join(ann_path, p) for p in os.listdir(ann_path) if p.endswith('.parquet')]
    chunk_size = len(files) // num_threads + int(bool(len(files) % num_threads)) # 100 processes
    chunks = [files[i: i+chunk_size] for i in range(0, len(files), chunk_size)]
    results, tmpfs, threads = [], [], []
    # pool = Pool(processes=num_workers)
    for i in range(len(chunks)):
        tmpfs.append(f'_{i}.'.join(save_path.split('.')))
        # results.append(pool.apply_async(_convert_obelicts_2pretrain,args=(chunks[i], img_dir, tmpfs[-1])))
        threads.append(threading.Thread(target=_convert_obelicts_2pretrain, args=(chunks[i], img_dir, tmpfs[-1])))
    for i in range(len(chunks)):
        threads[i].start()
    for i in range(len(chunks)):
        threads[i].join()
    # pool.close(); pool.join()
    # results = list(itertools.chain(*[rt.get() for rt in results]))
    # Merge into fewer files
    if len(tmpfs) > nmerge:
        full_fs, full_lines = [], []
        for i, files in enumerate([tmpfs[_: _+30] for _ in range(0, num_threads, 30)]):
            print(f"Merging into {i} ..")
            lines = [line for f in files for line in open(f)]
            full_lines.extend(lines)
            os.system(f"rm {' '.join(files)}")
            full_fs.append(f'_full_{i}.'.join(save_path.split('.')))
            with open(full_fs[-1], 'w') as f:
                f.writelines(lines)
    else:
        full_lines = [line for f in tmpfs for line in open(f)]
            
    # Filter to save different rounds
    print(f"Filtering to save files with different rounds ..")
    gt2out, gt3out, gt4out= [open(f'_gt{i}.'.join(save_path.split('.')), 'w') for i in range(2,5)]
    for line in tqdm(full_lines, desc=f"Filtering into different level"):
        ex = json.loads(line)
        if len(ex['image']) >= 4: gt2out.write(line)
        if len(ex['image']) >= 6: gt3out.write(line)
        if len(ex['image']) >= 8: gt4out.write(line)


def _repair(ori_examples):
    results = []
    for ex in tqdm(ori_examples):
        is_valid = []
        for img_path in ex['image']:
            if img_path is not None:
                is_valid.append(False)
                for i in range(10): # maximum retrials
                    try:
                        pil_img = Image.open(img_path)
                        if np.array(pil_img).dtype == np.uint8:
                            is_valid[-1] = True
                        break
                    except BaseException as e:
                        print(f"{e}")
                        pass
        if all(is_valid):
            results.append(ex)
        else:
            print(f"Filtered out a sample of {img_path} that cannot be loaded correctly ...")
    return results
    

def repair_obelics(ori_file, save_file=None):
    """ Filter out samples that cannot be loaded correctly.
    """
    results = []
    examples = [json.loads(line) for line in open(ori_file)]
    # Apply multiprocessing
    chunks = split_list(examples, 20)
    pool = ThreadPool(processes=20)
    for i in range(len(chunks)):
        results.append(pool.apply_async(_repair,args=(chunks[i], )))
    pool.close(); pool.join()
    results = list(itertools.chain(*[rt.get() for rt in results]))
    # Save
    save_file = save_file if save_file is not None else ori_file
    with open(save_file, 'w') as f:
        for ex in results:
            f.write(json.dumps(ex)+'\n')
        
if __name__ == '__main__':

    convert_obelicts_2pretrain(
        ann_path='Obelics/',
        img_dir='Obelics/images',
        save_path='obelics/obelics.jsonl'
    )

    repair_obelics('obelics/obelics_gt2.jsonl', 'obelics/obelics_gt2_repair.jsonl')
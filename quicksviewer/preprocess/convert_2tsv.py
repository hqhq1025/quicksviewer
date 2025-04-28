from torch.utils.data import Dataset, DataLoader, IterableDataset
import json
import time
import os
import os.path as op
import numpy as np
import base64
from tqdm import tqdm
import errno
from decord import VideoReader, cpu
import sys, io
from PIL import Image
import glob
import math
import pyarrow.parquet as pq
import pandas as pd
# from lightning.data import optimize, StreamingDataset

from quicksviewer.utils.data_util import uniform_sample
from quicksviewer.utils.mm_utils import is_multimodal

def save_update(save_results, save_file):
    if save_file.endswith('.jsonl'):
        with open(save_file+'.tmp', 'w') as f:
            f.writelines([json.dumps(ex)+'\n' for ex in save_results])
        os.system(f"cat {save_file}.tmp >> {save_file}; rm {save_file}.tmp")
    elif save_file.endswith('.json'):
        save_js = json.load(open(save_file)) if os.path.exists(save_file) else {}
        if isinstance(save_results, dict):
            save_js.update(save_results)
        elif isinstance(save_results, list):
            save_js.extend(save_results)
        json.dump(save_js, open(save_file, 'w'))

def tsv_writer(values, tsv_file, sep='\t'):
    os.makedirs(op.dirname(tsv_file), exist_ok=True)
    lineidx_file = op.splitext(tsv_file)[0] + '.lineidx'
    lastidx_file = lineidx_file.replace(".lineidx", ".last")
    idx = 0
    if os.path.exists(lineidx_file):
        last_line = list(open(lastidx_file, "r").readlines())[-1].strip()
        idx = int(last_line)
    # tsv_file_tmp = tsv_file
    # lineidx_file_tmp = lineidx_file
    tsv_file_tmp = tsv_file + '.tmp'
    lineidx_file_tmp = lineidx_file + '.tmp'
    with open(tsv_file_tmp, 'w') as fp, open(lineidx_file_tmp, 'w') as fpidx:
        assert values is not None
        for value in values:
            assert value is not None
            # this step makes sure python2 and python3 encoded img string are the same.
            # for python2 encoded image string, it is a str class starts with "/".
            # for python3 encoded image string, it is a bytes class starts with "b'/".
            # v.decode('utf-8') converts bytes to str so the content is the same.
            # v.decode('utf-8') should only be applied to bytes class type.
            value = [v if type(v)!=bytes else v.decode('utf-8') for v in value]
            v = '{0}\n'.format(sep.join(map(str, value)))
            fp.write(v)
            fpidx.write(str(idx) + '\n')
            idx = idx + len(v)
    with open(lastidx_file, "w") as fplast:
        fplast.write(str(idx) + '\n')
    # os.rename(tsv_file_tmp, tsv_file)
    # os.rename(lineidx_file_tmp, lineidx_file)
    os.system(f"cat {tsv_file_tmp} >> {tsv_file}"); os.remove(tsv_file_tmp)
    os.system(f"cat {lineidx_file_tmp} >> {lineidx_file}"); os.remove(lineidx_file_tmp)



def save_2tsv_dataset(examples, save_dir):
    """
      examples: [{
      'filename' or 'video' or 'image': str,
      'images': [base64],
      'conversations': [dict], }]
    """
    os.makedirs(save_dir, exist_ok=True)
    f = op.join(save_dir, 'imgs.lineidx')
    curid = 0 if not op.exists(f) else len(open(f).readlines()) # lineidx index starts with 0
    
    lidx, tsv, chat, fname2idx, len_dic = curid, [], [], {}, {}
    for ex in tqdm(examples):
        if len(ex) == 0 or len(ex.get('images', []))==0:
            continue
        filename = 'filename'
        if 'video' in ex: filename = 'video'
        elif 'image' in ex: filename = 'image'
        
        tsv.append([f"{ex[filename]}", json.dumps(ex['images'])])
        fname2idx[ex[filename]] = lidx
        len_dic[ex[filename]] = len(ex['images'])
        # if 'conversations' in ex:
        #     chat.append(ex['conversations'])
        ex.pop('images') # save exclude images
        chat.append(ex)
        lidx += 1
    save_update(chat, op.join(save_dir, 'chat.jsonl'))
    save_update(fname2idx, op.join(save_dir, 'fname2idx.json'))
    save_update(len_dic, op.join(save_dir, 'len_dic.json'))
    tsv_writer(tsv, op.join(save_dir,"imgs.tsv"))


def align_tsv_datasets(save_dir):
    """ Algin all data to be same number of samples according to `imgs.tsv`.
    """
    print(f"Begin aligning files for: {save_dir}")
    with open(op.join(save_dir, 'imgs.lineidx')) as f:
        tot_lines = len(f.readlines())
    # Align
    lineidx, fname2idx, invalid_tsv_lines = [], {}, []
    i, idx, last = 0, 0, 0
    with open(op.join(save_dir, 'imgs.tsv')) as f, open(op.join(save_dir, 'imgs_tmp.tsv'), 'w') as fout:
        for line in tqdm(f, total=tot_lines): # approximate tqdm number
            try:
                name, imgs = line.strip().split('\t')
                assert is_multimodal(name) # check fname
                pil_imgs = [Image.open(io.BytesIO(base64.b64decode(img))) for img in eval(imgs)]
                assert len(pil_imgs) > 0  # check imgs
                fout.write(line) # update imgs.tsv
                lineidx.append(idx)
                fname2idx[name] = i
                i += 1
                idx += len(line)
                last = idx
            except:
                invalid_tsv_lines.append((i, idx))
                print(f"Invalid tsv line: {(i, idx)}")
    os.rename(op.join(save_dir, 'imgs_tmp.tsv'), op.join(save_dir, 'imgs.tsv'))

    # Save
    new_chats = [_l for _l in open(op.join(save_dir, 'chat.jsonl')) if json.loads(_l)['video'] in fname2idx]
    assert len(new_chats) == len(fname2idx)
    print(f"Align fname2idx.json succeed.")
    # Save
    with open(op.join(save_dir, 'imgs.lineidx'), 'w') as f:
        for line in lineidx: f.write(str(line) + '\n')
    with open(op.join(save_dir, 'fname2idx.json'), 'w') as f:
        json.dump(fname2idx, f)
    with open(op.join(save_dir, 'chat.jsonl'), 'w') as f:
        for line in new_chats: f.write(line)
    with open(op.join(save_dir, 'imgs.last'), 'w') as f:
        f.write(str(last) + '\n')

def get_tsv_processed(save_dir):
    if not op.exists(op.join(save_dir, 'fname2idx.json')):
        return []
    fnames = list(json.load(open(op.join(save_dir, 'fname2idx.json'))).keys())
    # fnames = [f.split('_') for f in fnames][0]
    return fnames



class LLaVAParquetsDataset(IterableDataset):
    def __init__(self, parquets_dir):
        self.parquets = list(glob.glob(op.join(parquets_dir, '*.parquet')))
    
    def __iter__(self,):
        for parquet_file in self.parquets:
            parquet = pq.ParquetFile(parquet_file)
            # Process per batch to reduce RAM usage
            for batch in parquet.iter_batches(batch_size=32):
                df = batch.to_pandas()
                # df['image'] = df['image'].apply(lambda x: Image.open(io.BytesIO(x)))
                for row in df.itertuples():
                    # sample = (row.id, row.timestamp.strftime('%m/%d/%Y %I:%M %p'), row.description, row.image)
                    img_path = op.join(parquet_file, row.IMAGE_ID)
                    conv = {'conversations': json.loads(row.TEXT), 'image': img_path}
                    # sample = (conv, img_path, [str(base64.b64encode(row.BUFFER), encoding='utf-8')])
                    sample = {
                        'filename': img_path,
                        'images': [str(base64.b64encode(row.BUFFER), encoding='utf-8')],
                        'conversations':conv,
                    }
                    yield sample # -> encode the sample into binary chunks



    


if __name__ == '__main__':
    #  LLaVA-Pretrain
    save_dir = "/data/cc3m595"
    os.makedirs(save_dir, exist_ok=True)
    dataset = LLaVAParquetsDataset(parquets_dir="CC3M-Concept-balanced-595K/parquets")
    dataloader = DataLoader(dataset, num_workers=40, batch_size=50000, shuffle=False, collate_fn=lambda k: k)
    lidx, to_save, chat, fname2idx, len_dic = 0, [], [], {}, {}
    for batch in dataloader:
        save_2tsv_dataset(batch, save_dir)
        
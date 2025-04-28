# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.
import base64
import json
import logging
import os, re
import os.path as op
import sys
import numpy as np
import io
import cv2
import math
from decord import VideoReader, cpu
from PIL import Image


def find_consecutive_segments(A, label='video'):
    """ find consecutive segments where the values in each segment equals to `label`.
        e.g., [1,0,0,1,1,0,0,1] --> [(0,3), (3,4), (4,7), (7,8)]
        e.g., ['image', 'video', 'video', 'image'] --> [(1,3)]
    """
    segments = []
    pre, cur = 0, 0
    while cur < len(A):
        if A[pre] == A[cur] == label:
            cur += 1
        else:
            if cur > pre:
                segments.append((pre, cur))
            pre = cur = cur+1
    if cur > pre:
        segments.append((pre, cur))
    return segments

def uniform_sample(l, n):
    gap = len(l) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    idxs = sorted(set(idxs))
    return [l[i] for i in idxs], idxs

def split_list(lst, n):
    length = len(lst)
    size = math.ceil(length / n)
    return [lst[i * size:(i + 1) * size] for i in range(n)]


def opencv_extract_frames_fps(vpath, num_frames=None, fps=1.0, start_sec=None, end_sec=None, to_base64=False, to_pilimg=True):
    """ Evenly sampling 'num_frames' frames according to given fps if 'num_frames' is given,
          elsewise sampling all frames according to 'fps'.
    """
    fcap = cv2.VideoCapture(vpath)
    FPS = fcap.get(cv2.CAP_PROP_FPS)
    FCOUNT = fcap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = min(fps, FPS)
    # W, H = int(fcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(fcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_idx = round(FPS * start_sec) if start_sec else 0
    end_idx = round(FPS * end_sec) if end_sec else int(FCOUNT -1)
    # frame_idx = [i for i in range(start_idx, end_idx+1, round(FPS/fps))]
    frame_idx = [int(i+ii*FPS/fps+FPS/fps/2) for i in range(start_idx, end_idx+1, round(FPS)) for ii in range(round(fps))]
    frame_idx = uniform_sample(frame_idx, num_frames)[0] if num_frames is not None else frame_idx

    frame = fcap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx[0]))
    suc, frame = fcap.read()
    res_frames, timestamps, video_bytes = [], [], []
    ii = frame_idx[0]
    while suc and ii <= frame_idx[-1]:
        if ii in frame_idx:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # add this to make consistent with PIL.Image
            if to_base64:
                imgByteArr = io.BytesIO()
                Image.fromarray(frame).convert('RGB').save(imgByteArr, format='JPEG')
                img_bytes = imgByteArr.getvalue()
                encode_img = base64.b64encode(img_bytes)
                encode_img = str(encode_img, encoding='utf-8')
                video_bytes.append(encode_img)
            if to_pilimg:
                frame = Image.fromarray(frame)
            res_frames.append(frame)
            timestamps.append(round(ii/FPS, 1))
        suc, frame = fcap.read()
        ii += 1
    fcap.release()
    if not to_base64:
        return res_frames, timestamps
    else:
        return res_frames, timestamps, video_bytes



def decord_extract_frames_fps(vpath, num_frames=None, fps=1.0, start_sec=None, end_sec=None, to_base64=False):
    """ Sampling all frames according to given fps.
    """
    vr = VideoReader(vpath, ctx=cpu(0), num_threads=1)
    FPS, FRAME_COUNT = vr.get_avg_fps(), len(vr)
    fps = min(fps, FPS)

    start_idx = round(FPS * start_sec) if start_sec else 0
    end_idx = round(FPS * end_sec) if end_sec else FRAME_COUNT -1

    # frame_idx = [i for i in range(start_idx, end_idx+1, round(FPS / fps))]
    frame_idx = [int(i+ii*FPS/fps+FPS/fps/2) for i in range(start_idx, end_idx+1, round(FPS)) for ii in range(round(fps))]
    frame_idx= uniform_sample(frame_idx, num_frames)[0] if num_frames is not None else frame_idx
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    video = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in spare_frames]
    timestamps = [round(fid/FPS,1) for fid in frame_idx]
    video_bytes = []
    if to_base64:
        video_bytes = []
        for img in video:
            imgByteArr = io.BytesIO()
            img.save(imgByteArr, format='JPEG')
            img_bytes = imgByteArr.getvalue()
            encode_img = base64.b64encode(img_bytes)
            encode_img = str(encode_img, encoding='utf-8')
            video_bytes.append(encode_img)
    if not to_base64:
        return video, timestamps
    else:
        return video, timestamps, video_bytes




def generate_lineidx_file(filein, idxout):
    idxout_tmp = idxout + '.tmp'
    with open(filein, 'r') as tsvin, open(idxout_tmp,'w') as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0
        while fpos!=fsize:
            tsvout.write(str(fpos)+"\n")
            tsvin.readline()
            fpos = tsvin.tell()
    os.rename(idxout_tmp, idxout)

class TSVReader(object):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.data_map = {
            "sharegpt4video": json.load(open(os.path.join(self.img_dir,"ShareGPT4Video/fname2idx.json"))),
            "activitynet_caption": json.load(open(os.path.join(self.img_dir,"ANetCaptions/fname2idx.json"))),
            "youcook2": json.load(open(os.path.join(self.img_dir,"VideoChat2IT/YouCook2/fname2idx.json"))),
        }
        self.tsv_dic = {
            "sharegpt4video": TSVFile(os.path.join(self.img_dir, "ShareGPT4Video/imgs.tsv")),
            "activitynet_caption": TSVFile(os.path.join(self.img_dir, "ANetCaptions/imgs.tsv")),
            "youcook2": TSVFile(os.path.join(self.img_dir, "VideoChat2IT/YouCook2/imgs.tsv")),
        }

    def get(self, fname):
        dataset_name = None
        
        if 'sharegpt4video' in fname:
            dataset_name = 'sharegpt4video'
        elif 'YouCook2' in fname:
            dataset_name = 'youcook2'
        else:
            return None

        idx = self.data_map[dataset_name].get(fname, None) # Default
        if dataset_name == "sharegptvideo":
            fname = fname.replace("ShareGPTvideo_train_video_and_instruction/train_300k", "ShareGPTvideo")
            idx = self.data_map[dataset_name].get(fname, None)
        else:
            idx = self.data_map[dataset_name].get(fname, None)

        if dataset_name in ["sharegpt4video", 'activitynet_caption', "youcook2"]:
            if idx is None:
                print("idx is None")
                print("dataset name:",dataset_name)
                print("fname:",fname)
                return None
            vid_b64 = self.tsv_dic[dataset_name].seek(idx)[1].strip()
            vid_b64 = json.loads(vid_b64)
            vid_bytes = [base64.b64decode(x) for x in vid_b64]
            return vid_bytes

        img_b64 = self.tsv_dic[dataset_name].seek(idx)[1].strip()
        img_bytes = base64.b64decode(img_b64)
        return img_bytes


class TSVFile(object):
    def __init__(self, tsv_file, generate_lineidx=False):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx'
        self._fp = None
        self._lineidx = None
        # the process always keeps the process which opens the file.
        # If the pid is not equal to the currrent pid, we will re-open the file.
        self.pid = None
        # generate lineidx if not exist
        if not op.isfile(self.lineidx) and generate_lineidx:
            generate_lineidx_file(self.tsv_file, self.lineidx)

    def __del__(self):
        if self._fp:
            self._fp.close()

    def __str__(self):
        return "TSVFile(tsv_file='{}')".format(self.tsv_file)

    def __repr__(self):
        return str(self)

    def num_rows(self):
        self._ensure_lineidx_loaded()
        return len(self._lineidx)

    def seek(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        try:
            pos = self._lineidx[idx]
        except:
            logging.info('{}-{}'.format(self.tsv_file, idx))
            raise
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]


    def __getitem__(self, index):
        return self.seek(index)

    def __len__(self):
        return self.num_rows()

    def _ensure_lineidx_loaded(self):
        if self._lineidx is None:
            logging.info('loading lineidx: {}'.format(self.lineidx))
            with open(self.lineidx, 'r') as fp:
                self._lineidx = [int(i.strip()) for i in fp.readlines()]

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

        if self.pid != os.getpid():
            logging.info('re-open {} because the process id changed'.format(self.tsv_file))
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

    def gen_fname2idx(self):
        dic = {}
        total_len = len(self)
        from tqdm import tqdm
        for i in tqdm(range(total_len)):
            data = self.seek(i)
            fname = data[0]
            dic[fname] = i
        return dic
    


if __name__ == '__main__':


    vpath = 'Activity/v_zyi4NVylqB0_1.mp4'

    from datetime import datetime
    stime_load = datetime.now()
    video1 = opencv_extract_frames_fps(vpath, None, 1)
    print(f"Time consuming OpenCV loading video2: {(datetime.now()-stime_load).total_seconds()}")

    stime_load = datetime.now()
    video2 = decord_extract_frames_fps(vpath, None, 1)
    print(f"Time consuming Decord loading video2: {(datetime.now()-stime_load).total_seconds()}")

    print(video1==video2)


import os
import os.path as op
import json
import itertools
from multiprocessing import Pool, Lock, cpu_count
import math
import tqdm
import cv2




def split_list(lst, n):
    length = len(lst)
    size = math.ceil(length / n)
    return [lst[i * size:(i + 1) * size] for i in range(n)]


def extract_clips_save(params, max_off=float('Inf'), local_rank=0):
    """ Extract specify clips from the given video according to the offsets and durations using cv2.
      Args:
        params: [(f_video, offset, duration, save_name), ...]
    """
    # if local_rank == 0:
    data = tqdm.tqdm(params)
        
    for (f_v, off, dur, s_name) in data:
        # check legal offset
        if off > max_off:
            raise IndexError("The offset is out of the frame numbers.")
        fcap = cv2.VideoCapture(f_v)
        FPS = fcap.get(cv2.CAP_PROP_FPS)
        FCOUNT = fcap.get(cv2.CAP_PROP_FRAME_COUNT)
        W, H = int(fcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(fcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
        frame = fcap.set(cv2.CAP_PROP_POS_FRAMES, off*FPS)
        # save_path = os.path.join(save_dir, "{}.avi".format(i+start_idx))

        writer = cv2.VideoWriter(s_name, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), FPS, (W, H))
        # writer = cv2.VideoWriter(s_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), FPS, (W, H))

        suc, frame = fcap.read()
        ii = 0
        while suc and ii<dur*FPS:
            writer.write(frame)
            suc, frame = fcap.read()
            ii += 1
        writer.release()
        fcap.release()
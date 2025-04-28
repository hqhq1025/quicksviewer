import os,sys
import json
import tqdm
from multiprocessing.pool import ThreadPool
from multiprocessing import Lock
import itertools
from pathlib import Path
import re
from collections import defaultdict

from quicksviewer.preprocess.utils import split_list
from quicksviewer.utils.data_util import opencv_extract_frames_fps
from quicksviewer.preprocess.convert_2tsv import save_2tsv_dataset
from quicksviewer.utils.mm_utils import is_video

class DatasetBase:
    def __init__(self, num_workers):
        self.num_workers = num_workers
    
    def _build_samples(self, samples):
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
            # pool = multiprocessing.Pool(processes=self.workers)
            pool = ThreadPool(processes=self.num_workers)
            result = [pool.apply_async(self._build_samples, args=[trunks[i]]) for i in range(self.num_workers)]
            pool.close(); pool.join()
            result = list(itertools.chain(*[rt.get() for rt in result]))
        # Save
        if self.save_results:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, f'{self.prefix}.jsonl'), 'w') as f:
                for ex in result:
                    f.write(json.dumps(ex)+'\n')
        print(f"Completed to process {self.ann_path}")

class TextVR(DatasetBase):
    def __init__(self, ann_path="VideoChat2-IT/video/caption/textvr/train.json",
                 instr_path="VideoChat2-IT/video/caption/textvr/instructions.json",
                 video_dir="TextVR/TextVR_data/Video/",
                 output_dir='vid_train_datasets/video_data/VideoChat2IT/TextVR',
                 num_workers=1):
        super().__init__(num_workers)
        self.ann_path = ann_path
        self.instr_path = instr_path
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.prefix = 'VideoChat2IT_TextVR'
        self.save_results = True
    
    def _build_samples(self, samples):
        result = []
        for i, ex in tqdm.tqdm(enumerate(samples), desc="processing TextVR"):
            cat = ex['video'].split('/')[0]
            vpath = os.path.join(self.video_dir, ex['video'])
            assert os.path.exists(vpath), f"The video path does not exist: {vpath}"
            for qa in ex['QA']:
                result.append({
                    'uid': f'{self.prefix}_{cat}_{i}',
                    'video': vpath,
                    'conversations': [{'from':'human', 'value':qa['i']}, {'from':'gpt','value':qa['a']}],
                    'data_type': 'conversations',
                })
        return result


class YouCook2(DatasetBase):
    def __init__(self,
                ann_path="VideoChat2-IT/video/caption/youcook2/train.json",
                instr_path="VideoChat2-IT/video/caption/youcook2/instructions.json",
                ori_ann_path='YouCook2_train/annotations/youcookii_annotations_trainval.json',
                video_dir="YouCook2_train/raw_videos/",
                output_dir='vid_train_datasets/tsv_data/VideoChat2IT/YouCook2',
                num_workers=40):
        super().__init__(num_workers)
        self.ann_path = ann_path
        self.instr_path = instr_path
        self.ori_ann_path = ori_ann_path
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.prefix = 'VideoChat2IT_YouCook2'
        self.save_results = False
        # load original annotation
        with open(self.ori_ann_path) as f:
            self.ori_ann = json.load(f)['database']
        # resource manager
        self.lock = Lock()

    def _build_samples(self, samples):
        """ Convert to CSV.
        """
        for i, ex in tqdm.tqdm(enumerate(samples), desc="processing YouCook2"):
            result = []
            # Extract & Save clip
            split_id = re.match(r'.*?_(\d+)\..*',  Path(ex['video']).name).group(1)
            trainval, recipe, vname, splitname  =  Path(ex['video']).parts
            vpath = ''
            for fname in os.listdir(os.path.join(self.video_dir, trainval, recipe)):
                if vname in fname:
                    vpath = os.path.join(self.video_dir, trainval, recipe, fname)
            for qa in ex['QA']:
                ex = {
                    'uid': f'{self.prefix}_{trainval}_{recipe}_{vname}_{splitname}_{i}',
                    'conversations': [{'from':'human', 'value':qa['i']}, {'from':'gpt','value':qa['a']}],
                    'data_type': 'conversations',
                }
                # extract video clip
                ori_split_ann = self.ori_ann[vname]['annotations'][int(split_id)]
                assert ori_split_ann['id'] == int(split_id)
                start_sec, end_sec = ori_split_ann['segment']
                attempts, successful = 10, False
                while attempts > 0:
                    try:
                        res_frames, timestamps, video_bytes = opencv_extract_frames_fps(vpath,fps=1, start_sec=start_sec, end_sec=end_sec, to_base64=True, to_pilimg=False)
                        successful = True
                        break
                    except BaseException as e:
                        print(f"Failed to load video {vpath}, retrying ...")
                        attempts -= 1
                if not successful:
                    continue
                ex['images'] = video_bytes
                ex['video'] =  vpath.replace(vname, vname + f"_{end_sec}:{end_sec}")
                ex['timestamps'] = [max(0.0, round(t-start_sec,2)) for t in timestamps]
                result.append(ex)
            # Save
            self.lock.acquire()
            save_2tsv_dataset(result, self.output_dir)
            self.lock.release()
        return []



class Kinetics710(DatasetBase):
    def __init__(self, ann_path="VideoChat2-IT/video/classification/k710/train.json",
                 instr_path="VideoChat2-IT/video/classification/k710/instructions.json",
                 videos_dir={
                     'k400': 'VideoChat2-IT/raw_videos/k400',
                     'k600': 'VideoChat2-IT/raw_videos/k400',
                     'k700': 'VideoChat2-IT/raw_videos/k400'
                 },
                 output_dir='vid_train_datasets/video_data/VideoChat2IT/k710',
                 num_workers=1):
        """ Including K-400, K-600, K-700.
        """
        super().__init__(num_workers)
        self.ann_path = ann_path
        self.instr_path = instr_path
        self.videos_dir = videos_dir
        self.output_dir = output_dir
        self.prefix = 'VideoChat2IT_Kinetics710'
        self.save_results = True # whether to save result using self.build
        # load video paths
        self.vname2path = defaultdict(dict)
        for k,v in self.videos_dir.items():
            for root, folder, files in os.walk(v):
                for file in files:
                    if is_video(file):
                        self.vname2path[k][file] = os.path.join(root, file)
                    
    def _build_samples(self, samples):
        result = []
        for i, ex in tqdm.tqdm(enumerate(samples), desc="processing Kinetics710"):
            cat, vname = Path(ex['video']).parts[1], Path(ex['video']).name
            vpath = self.vname2path[cat].get(vname, '')
            if vpath == '':
                print(f"Not found for video path: {ex['video']}, skpping ...", flush=True)
                continue
            assert os.path.exists(vpath), f"The video path does not exist: {vpath}"
            for qa in ex['QA']:
                prompt = qa['i'] + ' ' + qa['q'] # Please xxx, Options: \n (A) ...
                response = qa['a'] # Answser: (A) xxx
                result.append({
                    'uid': f'{self.prefix}_{cat}_{i}',
                    'video': vpath,
                    'conversations': [{'from':'human', 'value':prompt}, {'from':'gpt','value':response}],
                    'data_type': 'conversations',
                })
        return result
    


class NextQA(DatasetBase):
    def __init__(self, ann_path="VideoChat2-IT/video/reasoning/next_qa/train.json",
                 instr_path="VideoChat2-IT/video/reasoning/next_qa/instructions.json",
                 video_dir='nextqa_videos',
                 output_dir='vid_train_datasets/video_data/VideoChat2IT/NextQA',
                 num_workers=1):
        """
        """
        super().__init__(num_workers)
        self.ann_path = ann_path
        self.instr_path = instr_path
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.prefix = 'VideoChat2IT_NextQA'
        self.save_results = True # whether to save result using self.build
                    
    def _build_samples(self, samples):
        result = []
        for i, ex in tqdm.tqdm(enumerate(samples), desc="processing NextQA"):
            cat, vname = Path(ex['video']).parts
            vpath = os.path.join(self.video_dir, vname)
            assert os.path.exists(vpath), f"The video path does not exist: {vpath}"
            for qa in ex['QA']:
                prompt = qa['i'] + ' ' + qa['q'] # Please xxx, Options: \n (A) ...
                response = qa['a'] # Answser: (A) xxx
                result.append({
                    'uid': f'{self.prefix}_{cat}_{i}',
                    'video': vpath,
                    'conversations': [{'from':'human', 'value':prompt}, {'from':'gpt','value':response}],
                    'data_type': 'conversations',
                })
        return result
    
class ClevrerQA(DatasetBase):
    def __init__(self, ann_path="VideoChat2-IT/video/reasoning/clevrer_qa/train.json",
                 instr_path="VideoChat2-IT/video/reasoning/clevrer_qa/instructions.json",
                 video_dir='clevrer/videos',
                 output_dir='vid_train_datasets/video_data/VideoChat2IT/ClevrerQA',
                 num_workers=1):
        """
        """
        super().__init__(num_workers)
        self.ann_path = ann_path
        self.instr_path = instr_path
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.prefix = 'VideoChat2IT_ClevrerQA'
        self.save_results = True # whether to save result using self.build
        # load video paths
        self.vname2path = {}
        for root, folder, files in os.walk(self.video_dir):
            for file in files:
                if is_video(file):  self.vname2path[file] = os.path.join(root, file)
                    
                    
    def _build_samples(self, samples):
        result = []
        for i, ex in tqdm.tqdm(enumerate(samples), desc="processing ClevrerQA"):
            cat, vname = 'ClevrerQA', ex['video']
            vpath = self.vname2path.get(vname, '')
            if vpath == '':
                print(f"Not found for video path: {ex['video']}", flush=True)
            assert os.path.exists(vpath), f"The video path does not exist: {vpath}"
            for qa in ex['QA']:
                prompt = qa['i'] + ' ' + qa['q'] # Please xxx, Options: \n (A) ...
                response = qa['a'] # Answser: (A) xxx
                result.append({
                    'uid': f'{self.prefix}_{cat}_{i}',
                    'video': vpath,
                    'conversations': [{'from':'human', 'value':prompt}, {'from':'gpt','value':response}],
                    'data_type': 'conversations',
                })
        return result


class ClevrerMC(DatasetBase):
    def __init__(self, ann_path="VideoChat2-IT/video/reasoning/clevrer_mc/train.json",
                 instr_path="VideoChat2-IT/video/reasoning/clevrer_mc/instructions.json",
                 video_dir='clevrer/videos',
                 output_dir='vid_train_datasets/video_data/VideoChat2IT/ClevrerMC',
                 num_workers=1):
        """
        """
        super().__init__(num_workers)
        self.ann_path = ann_path
        self.instr_path = instr_path
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.prefix = 'VideoChat2IT_ClevrerMC'
        self.save_results = True # whether to save result using self.build
        # load video paths
        self.vname2path = {}
        for root, folder, files in os.walk(self.video_dir):
            for file in files:
                if is_video(file):  self.vname2path[file] = os.path.join(root, file)
                    
    def _build_samples(self, samples):
        result = []
        for i, ex in tqdm.tqdm(enumerate(samples), desc="processing ClevrerMC"):
            cat, vname = 'ClevrerMC', ex['video']
            vpath = self.vname2path.get(vname, '')
            if vpath == '':
                print(f"Not found for video path: {ex['video']}", flush=True)
            assert os.path.exists(vpath), f"The video path does not exist: {vpath}"
            for qa in ex['QA']:
                prompt = qa['i'] + ' ' + qa['q'] # Please xxx, Options: \n (A) ...
                response = qa['a'] # Answser: (A) xxx
                result.append({
                    'uid': f'{self.prefix}_{cat}_{i}',
                    'video': vpath,
                    'conversations': [{'from':'human', 'value':prompt}, {'from':'gpt','value':response}],
                    'data_type': 'conversations',
                })
        return result



class EgoQA(DatasetBase):
    def __init__(self, ann_path="VideoChat2-IT/video/vqa/ego_qa/train.json",
                 instr_path="VideoChat2-IT/video/vqa/ego_qa/instructions.json",
                 video_dir='EgoQA/videos/split_videos/',
                 output_dir='vid_train_datasets/video_data/VideoChat2IT/EgoQA',
                 num_workers=1):
        """
        """
        super().__init__(num_workers)
        self.ann_path = ann_path
        self.instr_path = instr_path
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.prefix = 'VideoChat2IT_EgoQA'
        self.save_results = True # whether to save result using self.build
        # load video paths
        self.vname2path = {}
        for root, folder, files in os.walk(self.video_dir):
            for file in files:
                if is_video(file):  self.vname2path[file] = os.path.join(root, file)
                    
                    
    def _build_samples(self, samples):
        result = []
        for i, ex in tqdm.tqdm(enumerate(samples), desc="processing EgoQA"):
            cat, (vname, sname) = 'EgoQA', Path(ex['video']).parts
            vpath = os.path.join(self.video_dir, vname, sname)
            if not os.path.exists(vpath):
                print(f"The video path does not exist: {vpath}")
                continue
            for qa in ex['QA']:
                prompt = qa['i'] + ' ' + qa['q']
                response = qa['a']
                result.append({
                    'uid': f'{self.prefix}_{vname}_{sname}_{i}',
                    'video': vpath,
                    'conversations': [{'from':'human', 'value':prompt}, {'from':'gpt','value':response}],
                    'data_type': 'conversations',
                })
        return result



 
class TGIFFrameQA(DatasetBase):
    def __init__(self, ann_path="VideoChat2-IT/video/vqa/tgif_frame_qa/train.json",
                 instr_path="VideoChat2-IT/video/vqa/tgif_frame_qa/instructions.json",
                 video_dir='tgif/tgif_parts_release/gifs/',
                 output_dir='vid_train_datasets/video_data/VideoChat2IT/TGIF',
                 num_workers=1):
        """
        """
        super().__init__(num_workers)
        self.ann_path = ann_path
        self.instr_path = instr_path
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.prefix = 'VideoChat2IT_TGIFFrameQA'
        self.save_results = True # whether to save result using self.build
                                  
    def _build_samples(self, samples):
        result = []
        for i, ex in tqdm.tqdm(enumerate(samples), desc="processing TGIFFrameQA"):
            cat, vname = 'TGIFFrameQA', ex['video']
            vpath = os.path.join(self.video_dir, vname)
            assert os.path.exists(vpath), f"The video path does not exist: {vpath}"
            for qa in ex['QA']:
                result.append({
                    'uid': f'{self.prefix}_{cat}_{i}',
                    'video': vpath,
                    'conversations': [{'from':'human', 'value':qa['q']}, {'from':'gpt','value':qa['a']}],
                    'data_type': 'conversations',
                })
        return result



class TGIFTransitionQA(DatasetBase):
    def __init__(self, ann_path="VideoChat2-IT/video/vqa/tgif_transition_qa/train.json",
                 instr_path="VideoChat2-IT/video/vqa/tgif_transition_qa/instructions.json",
                 video_dir='tgif/tgif_parts_release/gifs/',
                 output_dir='vid_train_datasets/video_data/VideoChat2IT/TGIF',
                 num_workers=1):
        """
        """
        super().__init__(num_workers)
        self.ann_path = ann_path
        self.instr_path = instr_path
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.prefix = 'VideoChat2IT_TGIFTransitionQA'
        self.save_results = True # whether to save result using self.build
                                  
    def _build_samples(self, samples):
        result = []
        for i, ex in tqdm.tqdm(enumerate(samples), desc="processing TGIFTransitionQA"):
            cat, vname = 'TGIFTransitionQA', ex['video']
            vpath = os.path.join(self.video_dir, vname)
            assert os.path.exists(vpath), f"The video path does not exist: {vpath}"
            for qa in ex['QA']:
                prompt = qa['i'] + ' ' + qa['q']
                result.append({
                    'uid': f'{self.prefix}_{cat}_{i}',
                    'video': vpath,
                    'conversations': [{'from':'human', 'value':prompt}, {'from':'gpt','value':qa['a']}],
                    'data_type': 'conversations',
                })
        return result
    

class ShareGPTVideoVQA(DatasetBase):
    def __init__(self, ann_path="VideoChat2-IT/video/vqa/sharegptvideo/train_240k.json",
                 instr_path="VideoChat2-IT/video/vqa/sharegptvideo/instructions.json",
                 video_dir='ShareGPTVideo/activitynet_train/',
                 output_dir='vid_train_datasets/video_data/VideoChat2IT/ShareGPTVideo',
                 num_workers=1):
        """
        """
        super().__init__(num_workers)
        self.ann_path = ann_path
        self.instr_path = instr_path
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.prefix = 'VideoChat2IT_ShareGPTVideoVQA'
        self.save_results = True # whether to save result using self.build
                                  
    def _build_samples(self, samples):
        result = []
        for i, ex in tqdm.tqdm(enumerate(samples), desc="processing ShareGPTVideoVQA"):
            cat, vname = 'ShareGPTVideoVQA', ex['video'] + '.mp4'
            vpath = os.path.join(self.video_dir, vname)
            if not os.path.exists(vpath):
                print(f"Not found for video path: {vpath}")
                continue
            for qa in ex['QA']:
                prompt = qa['i'] + ' ' + qa['q']
                result.append({
                    'uid': f'{self.prefix}_{cat}_{i}',
                    'video': vpath,
                    'conversations': [{'from':'human', 'value':prompt}, {'from':'gpt','value':qa['a']}],
                    'data_type': 'conversations',
                })
        return result


class ShareGPTVideoCaption(DatasetBase):
    def __init__(self, ann_path="VideoChat2-IT/video/caption/sharegptvideo/train_300k.json",
                 instr_path="VideoChat2-IT/video/caption/sharegptvideo/instructions.json",
                 video_dir='ShareGPTVideo/activitynet_train/',
                 output_dir='vid_train_datasets/video_data/VideoChat2IT/ShareGPTVideo',
                 num_workers=1):
        """
        """
        super().__init__(num_workers)
        self.ann_path = ann_path
        self.instr_path = instr_path
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.prefix = 'VideoChat2IT_ShareGPTVideoCaption'
        self.save_results = True # whether to save result using self.build
                                  
    def _build_samples(self, samples):
        result = []
        for i, ex in tqdm.tqdm(enumerate(samples), desc="processing ShareGPTVideoCaption"):
            cat, vname = 'ShareGPTVideoCaption', ex['video'] + '.mp4'
            vpath = os.path.join(self.video_dir, vname)
            if not os.path.exists(vpath):
                print(f"Not found for video path: {vpath}")
                continue
            for qa in ex['QA']:
                prompt = qa['i']
                result.append({
                    'uid': f'{self.prefix}_{cat}_{i}',
                    'video': vpath,
                    'conversations': [{'from':'human', 'value':prompt}, {'from':'gpt','value':qa['a']}],
                    'data_type': 'conversations',
                })
        return result
    


class VideoChat2(DatasetBase):
    def __init__(self, ann_path="VideoChat2-IT/video/conversation/videochat2/train.json",
                 instr_path="VideoChat2-IT/video/conversation/videochat2/instructions.json",
                 video_dir='VideoChat2_conversation_videos/videos',
                 output_dir='vid_train_datasets/video_data/VideoChat2IT/VideoChat2',
                 num_workers=1):
        """
        """
        super().__init__(num_workers)
        self.ann_path = ann_path
        self.instr_path = instr_path
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.prefix = 'VideoChat2IT_VideoChat2'
        self.save_results = True # whether to save result using self.build
                                  
    def _build_samples(self, samples):
        result = []
        for i, ex in tqdm.tqdm(enumerate(samples), desc="processing VideoChat2"):
            cat, vname = 'VideoChat2', ex['video']
            vpath = os.path.join(self.video_dir, vname)
            if not os.path.exists(vpath):
                print(f"Not found for video path: {vpath}")
                continue
            for qa in ex['QA']:
                prompt = qa['q']
                result.append({
                    'uid': f'{self.prefix}_{cat}_{i}',
                    'video': vpath,
                    'conversations': [{'from':'human', 'value':prompt}, {'from':'gpt','value':qa['a']}],
                    'data_type': 'conversations',
                })
        return result


class VideoChatGPT(DatasetBase):
    def __init__(self, ann_path="VideoChat2-IT/video/conversation/videochatgpt/train.json",
                 instr_path="VideoChat2-IT/video/conversation/videochatgpt/instructions.json",
                 video_dir='activitynet/train',
                 output_dir='vid_train_datasets/video_data/VideoChat2IT/VideoChatGPT',
                 num_workers=1):
        """
        """
        super().__init__(num_workers)
        self.ann_path = ann_path
        self.instr_path = instr_path
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.prefix = 'VideoChat2IT_VideoChatGPT'
        self.save_results = True # whether to save result using self.build
                    
    def _build_samples(self, samples):
        result = []
        for i, ex in tqdm.tqdm(enumerate(samples), desc="processing VideoChatGPT"):
            cat, vname = 'train', Path(ex['video']).name # actually train set
            vpath = ''
            for suf in ['.mp4', '.mkv']:
                if os.path.exists(os.path.join(self.video_dir, Path(vname).with_suffix(suf))):
                    vpath = os.path.join(self.video_dir, Path(vname).with_suffix(suf))
            if not os.path.exists(vpath):
                print(f"The video path does not exist: {vpath}")
                continue
            for qa in ex['QA']:
                prompt = qa['q']
                response = qa['a']
                result.append({
                    'uid': f'{self.prefix}_{cat}_{vname}_{i}',
                    'video': vpath,
                    'conversations': [{'from':'human', 'value':prompt}, {'from':'gpt','value':response}],
                    'data_type': 'conversations',
                })
        return result
    

if __name__ == '__main__':
    for dataset in [TextVR(), YouCook2(), Kinetics710(), NextQA(), ClevrerQA(), ClevrerMC(), EgoQA(), TGIFFrameQA(), TGIFTransitionQA(), ShareGPTVideoVQA(), ShareGPTVideoCaption(), VideoChat2(), VideoChatGPT()]:
        dataset.build()
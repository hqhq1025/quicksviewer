import torch
from transformers import AutoModel, AutoTokenizer
import os,sys, re
import json
import numpy as np
import string
import math
import tqdm
import glob
import itertools
import multiprocessing
from PIL import Image
from argparse import ArgumentParser
from decord import VideoReader, cpu
from pathlib import Path
from typing import Dict, List
import argparse
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader

from quicksviewer.utils.mm_utils import get_model_name_from_path
from quicksviewer.utils.utils import disable_torch_init
from quicksviewer.model.builder import load_pretrained_model
from quicksviewer.utils.data_util import opencv_extract_frames_fps
from quicksviewer.eval.utils import extract_characters_regex
from quicksviewer.eval.tools.llm_eval_qa import evaluate as llm_eval_func

class BaseDataset(Dataset):
    def __init__(self, data_path, video_dir, samples_range, video_nframes, video_fps):
        self.data_path = data_path
        self.video_dir = video_dir
        
        # ActivityNet-QA
        with open(data_path) as f:
            data_q = json.load(f)
        vname2path = {vname.split('.')[0]: os.path.join(video_dir, vname) for vname in os.listdir(video_dir)}
        self.dataset = [{
            'video_file':vname2path[ex['video_name']],
            'question': [ex['Q']] if 'Q' in ex else [ex['Q1'], ex['Q2']], # a list
            'question_id': i,
            'answer': ex['A']
        } for i, ex in enumerate(data_q)]

        self.samples_range = list(map(lambda x: int(x) if x!='-1' else len(self.dataset), samples_range.split('_')))
        self.dataset = self.dataset[self.samples_range[0] : self.samples_range[1]]
        self.video_nframes = video_nframes
        self.video_fps = video_fps

        
    def __len__(self,):
        return len(self.dataset)
        
    def __getitem__(self, index):
        item = self.dataset[index]
        try:
            video, timestamps = opencv_extract_frames_fps(item['video_file'], self.video_nframes, self.video_fps, to_pilimg=False)
            video = np.stack(video, axis=0)
        except BaseException as e:
            video = np.zeros([self.video_nframes, 384,384,3], dtype=np.uint8)
            timestamps = [0.0]
            print(f"Using dummy video for {item['video_file']}")

        ex = {
            'video': video,
            'timestamps': timestamps,
            'question': item['question'],
            'question_id': item['question_id'],
            'answer':  item['answer'],
        }
        return ex


    def calc_scores(self,combined_contents):
        score_sum = 0
        count = 0
        for key, result in combined_contents.items():
            count += 1
            score_match = result[0]['score']
            score = int(score_match)
            score_sum += score
        average_score = score_sum / count
        metrics = {'average_score': average_score}
        print(str(metrics))
        return metrics
    
    def evaluate(self, results, llm_eval_func, save_result_f, save_metrics_f, out_dir):
        pred_contents = []
        for ex in results:
            ex['video_name'] = ex['question_id']
            pred_contents.append(ex)

        for subsubset in self.prompts:
            system = self.prompts[subsubset]['system']
            input_text = self.prompts[subsubset]['input_text']
            calc_scores = self.calc_scores
            out_dir = out_dir + subsubset.split('_')[0]
            save_result_f = save_result_f + subsubset.split('_')[0]
            save_metrics_f = save_metrics_f + subsubset.split('_')[0]
            llm_eval_func(
                pred_path=None,
                pred_contents=pred_contents,
                output_dir=out_dir,
                save_result=save_result_f,
                save_metrics=save_metrics_f,
                num_tasks=10,
                system=system,
                input_text=input_text,
                calc_scores=calc_scores
            )


class GenericQa(BaseDataset):
    def __init__(self, data_path, video_dir, samples_range, video_nframes, video_fps,):
        super().__init__(data_path, video_dir, samples_range, video_nframes, video_fps)

        self.prompts = {
            'correctness_prompts' : {'system': "You are an intelligent chatbot designed for evaluating the factual accuracy of generative outputs for video-based question-answer pairs. "
                                "Your task is to compare the predicted answer with the correct answer and determine if they are factually consistent. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "- Focus on the factual consistency between the predicted answer and the correct answer. The predicted answer should not contain any misinterpretations or misinformation.\n"
                                "- The predicted answer must be factually accurate and align with the video content.\n"
                                "- Consider synonyms or paraphrases as valid matches.\n"
                                "- Evaluate the factual accuracy of the prediction compared to the answer.",
                    'input_text': "Please evaluate the following video-based question-answer pair:\n\n"
                                    "Question: {question}\n"
                                    "Correct Answer: {answer}\n"
                                    "Predicted Answer: {pred}\n\n"
                                    "Provide your evaluation only as a factual accuracy score where the factual accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of factual consistency. "
                                    "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the factual accuracy score in INTEGER, not STRING."
                                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                    "For example, your response should look like this: {{''score': 4.8}}."
            },

            'detailed_orientation_prompts' : {'system':  "You are an intelligent chatbot designed for evaluating the detail orientation of generative outputs for video-based question-answer pairs. "
                                                    "Your task is to compare the predicted answer with the correct answer and determine its level of detail, considering both completeness and specificity. Here's how you can accomplish the task:"
                                                    "------"
                                                    "##INSTRUCTIONS: "
                                                    "- Check if the predicted answer covers all major points from the video. The response should not leave out any key aspects.\n"
                                                    "- Evaluate whether the predicted answer includes specific details rather than just generic points. It should provide comprehensive information that is tied to specific elements of the video.\n"
                                                    "- Consider synonyms or paraphrases as valid matches.\n"
                                                    "- Provide a single evaluation score that reflects the level of detail orientation of the prediction, considering both completeness and specificity.",
                    'input_text': "Please evaluate the following video-based question-answer pair:\n\n"
                                    "Question: {question}\n"
                                    "Correct Answer: {answer}\n"
                                    "Predicted Answer: {pred}\n\n"
                                    "Provide your evaluation only as a detail orientation score where the detail orientation score is an integer value between 0 and 5, with 5 indicating the highest level of detail orientation. "
                                    "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the detail orientation score in INTEGER, not STRING."
                                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                    "For example, your response should look like this: {{''score': 4.8}}."
            },

            'context_prompts': {'system':   "You are an intelligent chatbot designed for evaluating the contextual understanding of generative outputs for video-based question-answer pairs. "
                                "Your task is to compare the predicted answer with the correct answer and determine if the generated response aligns with the overall context of the video content. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "- Evaluate whether the predicted answer aligns with the overall context of the video content. It should not provide information that is out of context or misaligned.\n"
                                "- The predicted answer must capture the main themes and sentiments of the video.\n"
                                "- Consider synonyms or paraphrases as valid matches.\n"
                                "- Provide your evaluation of the contextual understanding of the prediction compared to the answer.",
                    'input_text': "Please evaluate the following video-based question-answer pair:\n\n"
                                    "Question: {question}\n"
                                    "Correct Answer: {answer}\n"
                                    "Predicted Answer: {pred}\n\n"
                                    "Provide your evaluation only as a contextual understanding score where the contextual understanding score is an integer value between 0 and 5, with 5 indicating the highest level of contextual understanding. "
                                    "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is contextual understanding score in INTEGER, not STRING."
                                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                    "For example, your response should look like this: {{''score': 4.8}}."
            }
        }


class TemporalQa(BaseDataset):
    def __init__(self, data_path, video_dir, samples_range, video_nframes, video_fps,):
        super().__init__(data_path, video_dir, samples_range, video_nframes, video_fps)
        
        self.prompts = {'temporal_prompts' : {'system': "You are an intelligent chatbot designed for evaluating the temporal understanding of generative outputs for video-based question-answer pairs. "
                                "Your task is to compare the predicted answer with the correct answer and determine if they correctly reflect the temporal sequence of events in the video content. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "- Focus on the temporal consistency between the predicted answer and the correct answer. The predicted answer should correctly reflect the sequence of events or details as they are presented in the video content.\n"
                                "- Consider synonyms or paraphrases as valid matches, but only if the temporal order is maintained.\n"
                                "- Evaluate the temporal accuracy of the prediction compared to the answer.",
                    'input_text': "Please evaluate the following video-based question-answer pair:\n\n"
                                "Question: {question}\n"
                                "Correct Answer: {answer}\n"
                                "Predicted Answer: {pred}\n\n"
                                "Provide your evaluation only as a temporal accuracy score where the temporal accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of temporal consistency. "
                                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the temporal accuracy score in INTEGER, not STRING."
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                "For example, your response should look like this: {{''score': 4.8}}."
            }}


class ConsistencyQa(BaseDataset):
    def __init__(self, data_path, video_dir, samples_range, video_nframes, video_fps,):
        super().__init__(data_path, video_dir, samples_range, video_nframes, video_fps)
        
        self.prompts = {'consistency_prompts' : {'system': "You are an intelligent chatbot designed for evaluating the consistency of generative outputs for similar video-based question-answer pairs. "
                            "You will be given two very similar questions, a common answer common to both the questions and predicted answers for the two questions ."
                            "Your task is to compare the predicted answers for two very similar question, with a common correct answer and determine if they are consistent. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the consistency between the two predicted answers and the correct answer. Both predicted answers should correspond to the correct answer and to each other, and should not contain any contradictions or significant differences in the conveyed information.\n"
                            "- Both predicted answers must be consistent with each other and the correct answer, in terms of the information they provide about the video content.\n"
                            "- Consider synonyms or paraphrases as valid matches, but only if they maintain the consistency in the conveyed information.\n"
                            "- Evaluate the consistency of the two predicted answers compared to the correct answer.",
                'input_text': "Please evaluate the following video-based question-answer pair:\n\n"
                            "Question 1: {question1}\n"
                            "Question 2: {question2}\n"
                            "Correct Answer: {answer}\n"
                            "Predicted Answer to Question 1: {pred1}\n"
                            "Predicted Answer to Question 2: {pred2}\n\n"
                            "Provide your evaluation only as a consistency score where the consistency score is an integer value between 0 and 5, with 5 indicating the highest level of consistency. "
                            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the consistency score in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {{''score': 4.8}}."
        }}



def run(args, dataloader, save_f=None) -> List:
    """
    Params:
        @vqas: [{'video_path', 'question_id', 'question', 'answer[optional]', ...}, ...]
    """
    # Load Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base,
        model_name, args.version, args.load_8bit, args.load_4bit,
        vpm_device=args.vpm_device, llm_device=args.llm_device,
        overwrite_config={'_attn_implementation':"flash_attention_2"}
    )

    # Evaluate
    results = []
    for batch in tqdm.tqdm(dataloader, desc=f"Evaluating on {str(dataloader.dataset)}"):
        for ex in batch:
            video, timestamps, question = ex['video'], ex['timestamps'], ex['question']
            res_list, question_list = [], ex['question']
            for question in question_list:
                question = question + '\n' + args.prompt if args.prompt else question
                msgs = [{'from':'human', 'value': [video] + [question], 'timestamps': timestamps},
                        {'from': 'gpt', 'value':''}]

                with torch.inference_mode():
                    try:
                        outputs = model.chat(
                            image=None,
                            msgs=msgs, # [{'from':'human'|'gpt','value':str|tuple, 'timestamps':list}]
                            modalities='video', # ['image', 'video', ..]
                            tokenizer=tokenizer,
                            image_processor=image_processor,
                            dtype=torch.float16,
                            llm_device=torch.device(f'cuda:{args.llm_device}') if args.vpm_device!=args.llm_device else None
                        )
                    except BaseException as e:
                        print(f"Error in model.chat: {e}, skipping the output ..")
                        outputs = ""
                res_list.append(outputs)
            if len(question_list) == 1:
                results.append({'pred': res_list[0], "question_id": ex['question_id'], 'question':question, 'answer': ex['answer']})
            else:
                results.append({'pred1': res_list[0], 'pred2': res_list[1], "question_id": ex['question_id'],  'question1':question_list[0],'question2':question_list[1], 'answer': ex['answer']})
    if save_f is not None:
        with open(save_f, 'w') as f:
            f.write(json.dumps(results))
    return results






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/quicksviewer-s3/checkpoint-15408")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--version", type=str, default='qwen2')
    parser.add_argument("--data_path", type=str, default='Video-ChatGPT/')
    parser.add_argument("--video_dir", type=str, default='Video-ChatGPT/activity/videos/')
    parser.add_argument("--samples_range", type=str, default='0_-1')
    parser.add_argument("--video_nframes", type=int, default=420)
    parser.add_argument("--video_fps", type=int, default=1)
    parser.add_argument("--vpm-device", type=int, default=0)
    parser.add_argument("--llm-device", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2) # set to 0.5 for video
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--prompt", type=str, default='')
    parser.add_argument("--task", type=str, default='videoChat_generic_qa')
    parser.add_argument("--save_dir", type=str, default='./output/results/')
    parser.add_argument("--calc_metrics", action='store_true')
    args = parser.parse_args()


    if args.task == "videoChat_generic_qa":
        ds_class = GenericQa
    elif args.task == "videoChat_consistency_qa":
        ds_class = ConsistencyQa
    else:
        ds_class = TemporalQa
    
    dataset = ds_class(
        data_path=os.path.join(args.data_path, args.task+".json"),
        video_dir=args.video_dir,
        samples_range=args.samples_range,
        video_nframes=args.video_nframes,
        video_fps=args.video_fps,
    )
    dataloader = DataLoader(dataset, batch_size=8, num_workers=4, collate_fn=lambda x:x)

    os.makedirs(args.save_dir, exist_ok=True)
    modelname = get_model_name_from_path(args.model_path)
    save_result = os.path.join(args.save_dir, f'{modelname}_{args.task}_result_fps={args.video_fps}_nframs={args.video_nframes}.json.{args.samples_range}')
    save_metrics = os.path.join(args.save_dir, f'{modelname}_{args.task}_metrics_fps={args.video_fps}_nframs={args.video_nframes}.json')
    save_gpt_result = os.path.join(args.save_dir, f'{modelname}_{args.task}_gptresult_fps={args.video_fps}_nframs={args.video_nframes}.json')
    save_dir = os.path.join(args.save_dir, f'{modelname}_{args.task}_gptresult_fps={args.video_fps}_nframs={args.video_nframes}')

    if not os.path.exists(save_result):
        results = run(args, dataloader)
        # save
        with open(save_result, 'w') as f:
            f.write(json.dumps(results))
        
    # Calculate metrics
    if args.calc_metrics: # Find all results splits
        results = []
        for f in glob.glob(".".join(save_result.split('.')[:-1]) + '*'):
            results.extend(json.load(open(f)))
        dataset.evaluate(results, llm_eval_func, save_gpt_result, save_metrics, save_dir)
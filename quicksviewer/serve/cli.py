import re
import argparse
import torch
from PIL import Image
from decord import VideoReader, cpu
import requests
from io import BytesIO
from transformers import TextStreamer
from copy import deepcopy
import numpy as np

from quicksviewer.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from quicksviewer.conversation import conv_templates, SeparatorStyle
from quicksviewer.model.builder import load_pretrained_model
from quicksviewer.utils.utils import disable_torch_init
from quicksviewer.utils.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from quicksviewer.utils.data_util import opencv_extract_frames_fps


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_video(video_path, nframs=None, fps=1):
    video, timestamps = opencv_extract_frames_fps(video_path, nframs, fps, to_pilimg=False)
    video = np.stack(video, axis=0)
    assert video.ndim==4
    return video, timestamps


def parse_dialogue(context_list, roles=['human', 'gpt']):
    """ Parse interleaved context into a multi-turn dialogue,
          where each turn starts with N images/videos followed with M prompt-reponse pairs.
    """
    i = 0
    dialogue, rou = [], []
    while i < len(context_list):
        if i==len(context_list)-1 or \
          not is_multimodal(context_list[i]) and is_multimodal(context_list[i+1]):
            rou.append(context_list[i])
            # parse into role value
            msgs, first_added = [{}], False
            for j,value in enumerate(rou):
                if is_multimodal(value) or not first_added:
                    ro = roles[0]
                    msgs[0]['from'] = ro
                    msgs[0]['value'] = msgs[0].get('value', []) + [value] # all imgs are put at begining
                    if not is_multimodal(value):
                        first_added= True
                else:
                    ro = roles[1]
                    if len(msgs) % 2 == 0:
                        ro = roles[0]
                    msgs.append({'from':ro, 'value': value})
            dialogue.extend(msgs)
            rou = []
        else:
            rou.append(context_list[i])
        i += 1
    return dialogue



from quicksviewer.utils.mm_utils import is_video, is_image, is_multimodal
def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base,
        model_name, args.version, args.load_8bit, args.load_4bit,
        vpm_device=args.vpm_device, llm_device=args.llm_device,
        overwrite_config={'_attn_implementation':"flash_attention_2"}
    )
    roles = ['human', 'gpt']

    ctx_dialogue = parse_dialogue(args.context) # [{'human': ['a.mp4', 'Hi'], 'gpt': 'hello'}]
    modality = 'video'
    msgs = deepcopy(ctx_dialogue)
    for rou in msgs:
        values = rou['value']
        values = values if isinstance(values, list) else [values]
        new_values, timestamps = [], None
        for v in values:
            if is_image(v):
                new_values.append(load_image(v))
                modality = 'image'
            elif is_video(v):
                frames, timestamps = load_video(v, args.video_nframes, args.video_fps)
                new_values.append(frames)
            else:
                new_values.append(v)
        rou['value'] = new_values
        rou['timestamps'] = timestamps

    use_kvcache = False
    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if not use_kvcache:
            assert msgs[-1]['from'] == roles[0], "The context must end with an image or a video."
            msgs[-1]['value'].append(inp) # Add current input prompt
        else:
            msgs = [{'from':roles[0], 'value':inp}]
        msgs.append({'from': roles[1], 'value':''}) # add the assistant flag

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.inference_mode():
            outputs = model.chat(
                image=None,
                msgs=msgs, # [{'from':'human'|'gpt','value':str|tuple, 'timestamps':list}]
                modalities=modality, # ['image', 'video', ..]
                tokenizer=tokenizer,
                image_processor=image_processor,
                dtype=torch.float16,
                llm_device=torch.device(f'cuda:{args.llm_device}') if args.vpm_device!=args.llm_device else None
            )
            print(outputs)
            msgs[-1]['value'] = outputs[0] # add model output as history
            msgs.append({'from': roles[0], 'value':[]})
        if args.debug:
            print("\n", {"prompt": inp, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/quicksviewer-s3/checkpoint-15408")
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--version", type=str, default='llama_3_chat')
    parser.add_argument("--version", type=str, default='qwen2')
    parser.add_argument("--context", nargs='+', default=['playground/demo/examples/tokyo_people.mp4'], help="The interleaved context, like [img,prompt,response,img,prompt,response, ..]")
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
    args = parser.parse_args()
    main(args)

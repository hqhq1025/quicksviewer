import gradio as gr
import os
import cv2
import torch
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from copy import deepcopy


from quicksviewer.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from quicksviewer.conversation import conv_templates, SeparatorStyle
from quicksviewer.model.builder import load_pretrained_model
from quicksviewer.utils.utils import disable_torch_init
from quicksviewer.utils.data_util import opencv_extract_frames_fps
from quicksviewer.utils.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, is_image, is_video, is_multimodal
from quicksviewer import conversation as conversation_lib
from quicksviewer.constants import DEFAULT_IMAGE_TOKEN
from quicksviewer.serve.cli import parse_dialogue


class InferenceDemo(object):
    def __init__(self,args,model_path,tokenizer, model, image_processor, context_len) -> None:
        disable_torch_init()

        self.tokenizer, self.model, self.image_processor, self.context_len = tokenizer, model, image_processor, context_len

        self.conversation = conversation_lib.default_conversation.copy()
        self.video_nframes = args.video_nframes
        self.video_fps = args.video_fps



def is_valid_video_filename(name):
    video_extensions = ['avi', 'mp4', 'mov', 'mkv', 'flv', 'wmv', 'mjpeg']
    
    ext = name.split('.')[-1].lower()
    
    if ext in video_extensions:
        return True
    else:
        return False


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            print('failed to load the image')
    else:
        print('Load image from local file')
        print(image_file)
        image = Image.open(image_file).convert("RGB")
        
    return image


def load_video(video_path, nframs=None, fps=1):
    video, timestamps = opencv_extract_frames_fps(video_path, nframs, fps, to_pilimg=False)
    video = np.stack(video, axis=0)
    assert video.ndim==4
    return video, timestamps

def clear_history(history):
    our_chatbot.conversation = conversation_lib.default_conversation.copy()
    return None


def clear_response(history):
    for index_conv in range(1, len(history)):
        # loop until get a text response from our model.
        conv = history[-index_conv]
        if not (conv[0] is None):
            break
    question = history[-index_conv][0]
    history = history[:-index_conv]
    return history, question


def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_message(history, message):
    # history=[]
    global our_chatbot
    if len(history)==0:
        our_chatbot = InferenceDemo(args,model_path,tokenizer, model, image_processor, context_len)
    
    print(f"[Debug add_message] history: {history}\n message: {message}")
    for x in message["files"]:
        history.append(((x), 'human'))
    if message["text"] is not None:
        history.append((message["text"], 'human'))
    return history, gr.MultimodalTextbox(value=None, interactive=False)



def bot(history):
    
    print(f"[Debug bot] history: {history}")
    context = []
    for x in history:
        context.append(x[0])
    assert history[-1][-1] == 'human'
    
    roles = ['human', 'gpt']
    ctx_dialogue = parse_dialogue(context) # [{'human': ['a.mp4', 'Hi'], 'gpt': 'hello'}]
    print(ctx_dialogue)
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
                frames, timestamps = load_video(v, our_chatbot.video_nframes, our_chatbot.video_fps)
                new_values.append(frames)
            else:
                new_values.append(v)
        rou['value'] = new_values
        rou['timestamps'] = timestamps

    use_kvcache = False
    if use_kvcache:
        assert msgs[-1]['from'] == roles[0], "The context must end with an image or a video."

    msgs.append({'from': roles[1], 'value':''}) # add the assistant flag
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    with torch.inference_mode():
        outputs = model.chat(
            image=None,
            msgs=msgs, # [{'from':'human'|'gpt','value':str|tuple, 'timestamps':list}]
            modalities=modality, # ['image', 'video', ..]
            tokenizer=tokenizer,
            image_processor=image_processor,
            dtype=torch.bfloat16,
            llm_device=torch.device(f'cuda:{args.llm_device}') if args.vpm_device!=args.llm_device else None
        )
        outputs = outputs[0]
        print(outputs)
        msgs[-1]['value'] = outputs # add model output as history
        msgs.append({'from': roles[0], 'value':[]})

    our_chatbot.conversation.append_message(roles[1], None)
    our_chatbot.conversation.messages[-1][-1] = outputs
   
    history.append((outputs, 'gpt'))
    return history


txt = gr.Textbox(
    scale=4,
    show_label=False,
    placeholder="Enter text and press enter.",
    container=False,
)


with gr.Blocks() as demo:
    # Informations
    title_markdown = ("""
        # Quicksviewer
        [[Project]](https://quicksviewer.github.io)  [[Code]](https://github.com/quicksviewer/quicksviewer) [[Model]](https://huggingface.co/qijithu/quicksviewer)
    """)
    tos_markdown = ("""
    ### TODO!. Terms of use
    By using this service, users are required to agree to the following terms:
    The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
    Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
    For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
    """)
    learn_more_markdown = ("""
    ### TODO!. License
    The service is a research preview, subject to the model [License](https://github.com/QwenLM/Qwen/blob/main/LICENSE) of Qwen or [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA. Please contact us if you find any potential violation.
    """)
    models = [
        "quicksviewer-8B",
    ]
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gr.Markdown(title_markdown)

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False
    )
    chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image","video"], placeholder="Enter message or upload file...", show_label=False)
    
    

    with gr.Row():
        upvote_btn = gr.Button(value="👍  Upvote", interactive=True)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=True)
        flag_btn = gr.Button(value="⚠️  Flag", interactive=True)
        #stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=True)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)

    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    chatbot.like(print_like_dislike, None, None)
    clear_btn.click(fn=clear_history, inputs=[chatbot], outputs=[chatbot], api_name="clear_all")
    with gr.Column():
        gr.Examples(examples=[
            [{"files": [f"{cur_dir}/examples/fangao3.jpeg",f"{cur_dir}/examples/fangao2.jpeg",f"{cur_dir}/examples/fangao1.jpeg"], "text": "Do you kown who draw these paintings?"}],
        ], inputs=[chat_input], label="Compare images: ")

demo.queue()
if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--server_name", default="0.0.0.0", type=str)
    argparser.add_argument("--port", default="6123", type=str)
    argparser.add_argument("--model-path", type=str, default="checkpoints/quicksviewer-s3/checkpoint-15408")
    argparser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--version", type=str, default='llama_3_chat')
    argparser.add_argument("--version", type=str, default='qwen2')
    argparser.add_argument("--context", nargs='+', default=['playground/demo/examples/tokyo_people.mp4'], help="The interleaved context, like [img,prompt,response,img,prompt,response, ..]")
    argparser.add_argument("--video_nframes", type=int, default=420)
    argparser.add_argument("--video_fps", type=int, default=1)
    argparser.add_argument("--vpm-device", type=int, default=0)
    argparser.add_argument("--llm-device", type=int, default=1)
    argparser.add_argument("--conv-mode", type=str, default=None)
    argparser.add_argument("--temperature", type=float, default=0.2)
    argparser.add_argument("--max-new-tokens", type=int, default=512)
    argparser.add_argument("--load-8bit", action="store_true")
    argparser.add_argument("--load-4bit", action="store_true")
    argparser.add_argument("--debug", action="store_true")
    
    args = argparser.parse_args()
    model_path = args.model_path
    filt_invalid="cut"
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base,
        model_name, args.version, args.load_8bit, args.load_4bit,
        vpm_device=args.vpm_device, llm_device=args.llm_device,
        overwrite_config={'_attn_implementation':"flash_attention_2"}
    )

    our_chatbot = None
    try:
        demo.launch(server_name=args.server_name, server_port=int(args.port),share=True)
    except Exception as e:
        args.port=int(args.port)+1
        print(f"Port {args.port} is occupied, try port {args.port}")
        demo.launch(server_name=args.server_name, server_port=int(args.port),share=True)

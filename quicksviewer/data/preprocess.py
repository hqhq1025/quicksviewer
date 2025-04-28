import copy
from typing import Dict, Sequence
import torch
import transformers
import numpy as np

from quicksviewer.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from quicksviewer.constants import DEFAULT_PATCH_START_TOKEN, DEFAULT_PATCH_END_TOKEN, DEFAULT_VIDEO_START_TOKEN, DEFAULT_VIDEO_END_TOKEN

from quicksviewer import conversation as conversation_lib
from quicksviewer.utils.mm_utils import tokenizer_image_token


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation



def preprocess_multimodal_image(
        sources: Sequence[str],
        data_args,
        pathnums_imgs,
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        imgid = 0
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].strip()
                if imgid >= len(pathnums_imgs):
                    break

                new_value = ""
                pre_idx = 0
                while imgid < len(pathnums_imgs):
                    replace_token = DEFAULT_IMAGE_TOKEN
                    if data_args.mm_use_patch_start_end:
                        replace_token = (DEFAULT_PATCH_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_PATCH_END_TOKEN)
                    replace_token = replace_token * pathnums_imgs[imgid]
                    if data_args.mm_use_im_start_end:
                        replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN

                    if "mmtag" in conversation_lib.default_conversation.version:
                        sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                                    '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')


                    cur_idx = sentence['value'].find(DEFAULT_IMAGE_TOKEN, pre_idx)
                    if cur_idx<0:
                        break
                    new_value += sentence['value'][pre_idx: cur_idx] + replace_token
                    pre_idx = cur_idx + len(DEFAULT_IMAGE_TOKEN)
                    imgid += 1

                new_value += sentence['value'][pre_idx:] 
                sentence["value"] = new_value

    return sources

from quicksviewer.constants import DEFAULT_THUMBNAIL_START_TOKEN, DEFAULT_THUMBNAIL_END_TOKEN
def preprocess_multimodal_video(
        sources: Sequence[str],
        data_args,
        nframes=64,
        frame_timestamps = [],
        is_cubing=False,
        add_thumbnail=False,
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    if len(frame_timestamps)==0:
        frame_timestamps = [""]*nframes
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                replace_token = ''.join([str(tmp) + DEFAULT_IMAGE_TOKEN for tmp in frame_timestamps])

                if add_thumbnail:
                    if data_args.mm_use_thumbnail_start_end:
                        replace_token = DEFAULT_THUMBNAIL_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_THUMBNAIL_END_TOKEN + replace_token
                    else:
                        replace_token = DEFAULT_IMAGE_TOKEN + replace_token

                if data_args.mm_use_video_start_end:
                    replace_token = DEFAULT_VIDEO_START_TOKEN +replace_token + DEFAULT_VIDEO_END_TOKEN
                    
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                                  '<Video>' + DEFAULT_IMAGE_TOKEN + '</Video>')
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources



def preprocess_llama_3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    no_system_prompt: bool = True,
    build_labels: bool = True,
) -> Dict:
    has_image = DEFAULT_IMAGE_TOKEN in sources[0][0]['value']
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    if no_system_prompt:
        conv.system = ""

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = None
    if build_labels:
        targets = input_ids.clone()
        assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

        # Mask targets
        sep = conv.sep + conv.roles[1]
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep)
            cut = 2 if no_system_prompt else 3
            re_rounds = [conv.sep.join(rounds[:cut])]  # system + user + gpt
            # for conv_idx in range(3, len(rounds), 2):
            for conv_idx in range(cut, len(rounds), 2):
                re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))  # user + gpt
            cur_len = 0
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(re_rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    # import ipdb; ipdb.set_trace()
                    print(f"WARNING: parts!=: {parts}")
                    break
                parts[0] += sep

                if has_image:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) 

                instruction_len = instruction_len-1 if i>0 else instruction_len
                round_len = round_len-1 if i>0 else round_len
                target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
                # cur_len += round_len+1 # +1 for skipping <|eot_id|>
                cur_len += round_len+1 if len(re_rounds)>1 else round_len # +1 for skipping <|eot_id|> at the end of each round

            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. {sources}" f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_qwen_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    # has_image: bool = False,
    no_system_prompt: bool = True,
    build_labels: bool = True,
) -> Dict:
    has_image = DEFAULT_IMAGE_TOKEN in sources[0][0]['value']
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    if no_system_prompt:
        conv.system = ""

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = None
    if build_labels:
        targets = input_ids.clone()
        assert conv.sep_style == conversation_lib.SeparatorStyle.QWEN2

        # Mask targets
        sep = conv.sep + conv.roles[1]
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep)
            cut = 2 if no_system_prompt else 3
            re_rounds = [conv.sep.join(rounds[:cut])]  # system + user + gpt
            # for conv_idx in range(3, len(rounds), 2):
            for conv_idx in range(cut, len(rounds), 2):
                re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))  # user + gpt
            cur_len = 0
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(re_rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    # import ipdb; ipdb.set_trace()
                    print(f"WARNING: parts!=: {parts}")
                    break
                parts[0] += sep

                if has_image:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids)
                target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len + len(tokenizer(conv.sep).input_ids) # Qiji: skip "<|im_end|>\n"

            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. {sources}" f" (ignored)")
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
        prompt: str = None,
        refine_prompt: bool = False,
        build_labels: bool = True,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3:
        return preprocess_llama_3(sources, tokenizer, no_system_prompt=True, build_labels=build_labels)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.QWEN2:
        return preprocess_qwen_2(sources, tokenizer, no_system_prompt=True, build_labels=build_labels)

    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers) # TODO for interleave

    return dict(input_ids=input_ids, labels=targets)

import os,sys
import re

# Copied from: https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/main/videollama2/eval/eval_video_mcqa_videomme.py
def extract_characters_regex(s, maximum=10):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
        "Answer:"
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    chars = [chr(ord('A')+_) for _ in range(maximum)]
    chars = "[{}]".format(''.join(chars))
    if len(s.split()) > 10 and not re.search(chars, s):
        return ""
    matches = re.search(chars, s)
    if matches is None:
        return ""
    return matches[0]


def extract_integers_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
        "Answer:"
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    matches = re.search(r'\d+', s)
    if matches is None:
        return ""
    return int(matches[0])
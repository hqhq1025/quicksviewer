import string
import random

IMG_REPLACE=['image', 'picture', 'photo']
VIDEO_REPLACE=['video', 'clip', 'media']


DESC_EN = [
    "What's shown in this {image}",
    'What do you think happened in this {image}?',
    'Can you explain the elements in the {image} you provided?',
    'What is this {image} about?',
    'A comprehensive and detailed analysis of the {image}.',
    'A detailed description of the given {image}.',
    "What's the key element in this {image}?",
    'Explaining the visual content of an {image} in detail.',
    'What do you see in this {image}?',
    'Can you describe the main features of this {image}?',
    'What happened at the {image}?',
    'This work depicts',
    'This {image} conveys the following information.',
    'Describe this {image}.',
    'From this {image}, we can see',
    'The elements in the {image} include:',
    'Describe the scene in this {image}.',
    'The {image} presents the following scenes.',
    'This {image} can be described as',
    'Explain this {image}.',
    'What does this {image} show?',
    'From this {image}, we can discover',
    'What does this {image} depict?',
    'What information does this {image} want to convey?',
    'This {image} displays',
    'What happens in this scenario?',
    'Describe the above {image}.',
    'The {image} shows the following scenes.',
    'What does this {image} describe?',
    'Thoroughly and meticulously analyze the {image}.'
]

DETAIL_DESC_VIDEO_EN = [
    'Please provide a detailed description of what happens in this video.',
    'Could you describe in detail the events that occur in this video?',
    'I would like a thorough account of the content in this video. Can you describe it for me?',
    'What are the specific details of what takes place in this video?',
    'Please elaborate on the sequence of events shown in this video.',
    'Can you summarize the actions and occurrences in this video in detail?',
    'I need a detailed breakdown of the happenings in this video. Can you help?',
    'What exactly is going on in this video? Please describe it.',
    "Provide a comprehensive description of the video's content, please.",
    "I'm looking for a detailed recount of the video's events. Can you tell me what happens?",
    "Could you walk me through what's shown in this video in a detailed manner?",
    "Could you furnish a detailed narrative of the events depicted in this video?",
    "I'm curious about the specifics of this video's content. Could you detail it for me?",
    "What is the detailed sequence of actions in this video?",
    "Will you provide an in-depth explanation of the events occurring in this video?",
    "I'd like to know the details of what's shown in the video. Can you elaborate?",
    "Please give me a detailed run-through of the video's storyline.",
    "Is it possible to get a detailed analysis of the events in this video?",
    "I need a detailed recount of the activities in the video. Can you help me with that?",
    "Could you provide a detailed breakdown of the video's sequence of events?",
    "I'm interested in a thorough description of the video's content. Would you mind detailing it for me?",
]



INSTRUCT_OPTIONS = [
    "Please select one correct answer from the following options based on the {image} content and the given question.",
    "Choose the correct answer from the provided options, in accordance with the {image}'s content and the specified question.",
    "Based on the {image} material and the question presented, pick the right option from the list below.",
    "Select the accurate response from the following choices, reflecting on the content of the {image} and the question at hand.",
    "From the options listed, identify the correct answer that corresponds to the {image} content and the posed question.",
    "Make your selection of the proper answer from these options, guided by the {image}'s information and the question provided.",
    "Choose the right answer among the options here, keeping in mind the details from the {image} and the question given.",
    "Select one valid answer from the options below, as per the content of the {image} and the question that was asked.",
    "Based on what you've seen in the {image} and the question asked, choose the correct option from the choices presented.",
]


def build_with_options(options, answer_position=None, question=None):
    """
      Return:
        instruction: a sentence.
        answer: an answer with options like: Anser: (1) xxx\n (2) xxx\n
    """
    # Build Instruction
    options_token = random.choice(["Options:"])
    split_token = random.choice(["\n"]) # or

    num_choices = list(range(1, len(options)+1)) # 1, 2, 3, ..
    num_choices_b = [f'({c})' for c in num_choices] # (1), (2), (3), ..
    lower_letter_choices = [f'({c})' for c in string.ascii_lowercase] # (a), (b), (c), ..
    upper_letter_choices = [f'({c})' for c in string.ascii_uppercase] # (A), (B), (C), ..
    op_choices = [f'Option {c}:' for c in num_choices] # Option 1:, Option 2:, ...
    choices_list = [('\n', num_choices, split_token), ('\n', num_choices_b, split_token), ('\n', lower_letter_choices, split_token), ('\n', upper_letter_choices, split_token), ('\n', op_choices, split_token)]
    tgt_choice = random.choice(choices_list)
    
    options_list = [f'{tgt_choice[1][i]}. {option}' for i, option in enumerate(options)]
    options_str = tgt_choice[2].join(options_list)
    options_str = f'{options_token}{tgt_choice[0]}{options_str}'

    instruction = random.choice(INSTRUCT_OPTIONS).format(image='video')
    if question is not None:
        instruction = instruction + ' Question: ' + question
    instruction = instruction + ' ' + options_str

    # Build answer
    answer = None
    if answer_position is not None:
        answer = 'Answer: ' + options_list[answer_position]
    return instruction, answer





if __name__ == '__main__':
    options = ["o1", "o2", "o3", "o4"]
    print(build_with_options(options, answer_position=1, question='Here is a question.'))
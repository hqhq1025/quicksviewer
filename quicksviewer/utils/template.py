
DESC_IMG = [
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
    'What happens in this {image}?',
    'Describe the above {image}.',
    'The {image} shows the following scenes.',
    'What does this {image} describe?',
    'Thoroughly and meticulously analyze the {image}.']


DESC_VID=[
    'Can you describe what happens in this {video}?',
    'What is depicted in this {video}?',
    'Could you summarize the content of this {video}?',
    'What occurs in this {video}?',
    'Please narrate the events shown in this {video}.',
    'What is the subject of this {video}?',
    'What scenes are shown in this {video}?',
    'Can you explain the action in this {video}?',
    'What does this {video} illustrate?',
    'What content is captured in this {video}?',
    'Could you detail what takes place in this {video}?',
   ' What is being portrayed in this {video}?',
    'Please provide a description of this {video}.',
    'What story does this {video} tell?',
    "What's happening in the {video}?",
    "Could you clarify what is shown in this {video}?",
    "What is the main focus of this {video}?",
    "Can you break down the events in this {video}?",
    "What events are recorded in this {video}?",
    "Please elaborate on the scenes in this {video}.",
]


import random
def random_one_desc(type='image', replace=None):
    source = DESC_IMG if type=='image' else DESC_VID
    desc = random.choice(source)
    if replace:
        desc = desc.format(image=replace) if type=='image' else desc.format(video=replace)
    return desc


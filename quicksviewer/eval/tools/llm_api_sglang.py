""" 
    Develop n SGLang servers first:  `CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --port 7890`
 --host 0.0.0.0
"""
import requests
import random

# The SGLang backend must has been started
class SGLangAPI():
    def __init__(self,
                 url= "http://127.0.0.1:7890/v1/chat/completions",
                 model= 'Qwen2.5-7B-Instruct'):
        # self.url =url
        self.urls = [f"http://127.0.0.1:{p}/v1/chat/completions" for p in [7890,7891,7892,7893]] # fixed
        self.model = model
        self.last = 0

    def ask_txt(self, txt):
        data = {
            'model': self.model,
            "messages": [{"role": "user", "content": txt}],
        }
        url = random.choice(self.urls)
        req = requests.post(url, json=data)
        parse = req.json()
        try:
            parse = parse['choices'][0]['message']['content']
        except:
            print(f"Faild to parse response from SGLang !!")
        return parse
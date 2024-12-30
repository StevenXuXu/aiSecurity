from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "/home/ubuntu/LLM/Llama-Guard-2-8B"
device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)


def moderate(chat):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    user_num_tokens = len(tokenizer.tokenize(chat[0]['content']))
    assistant_num_tokens = len(tokenizer.tokenize(chat[1]['content']))

    output = model.generate(
        input_ids=input_ids, 
        max_new_tokens=100, 
        pad_token_id=tokenizer.eos_token_id, 
        return_dict_in_generate=True, 
        output_scores=True
    )

    prompt_len = input_ids.shape[-1]
    generated_token_ids = output.sequences[0][prompt_len:]
    # print(generated_token_ids)
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_token_ids, skip_special_tokens=True)

    probs = torch.cat(output.scores).softmax(dim=-1)
    generated_token_probs, _ = torch.max(probs, dim=-1)

    # for token_id, token, prob in zip(generated_token_ids, generated_tokens, generated_token_probs):
    #     print(f'{token_id.item():<7} {token:<7} {prob.item():.4f}')

    response = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    # print(f"Generated response: {response!r}")
    return response, user_num_tokens, assistant_num_tokens


chat = [
    {"role": "user", "content": "写一首诗"},
    {"role": "assistant", "content": "花开，风舞自轻盈。清韵，不与春争名。"},
]

from llm_guard.input_scanners import PromptInjection
from llm_guard.input_scanners.prompt_injection import MatchType

scanner = PromptInjection(threshold=0.5, match_type=MatchType.FULL)

import os

# 定义目录路径
directory_path = '/home/ubuntu/Sensitive-lexicon/Vocabulary'
detect_path = '/home/ubuntu/normal-prompt-generation/output.txt'
save_path = '/home/ubuntu/normal-prompt-generation/output-filter.txt'

# 初始化一个空列表用于存储词汇
words = []
prompts = []
words_filename = ['暴恐', '反动', '色情']

# 遍历目录中的所有文件
for filename in os.listdir(directory_path):
    # 只处理以.txt结尾的文件
    if filename.endswith('.txt') and any(word in filename for word in words_filename):
        # 构建文件的完整路径
        file_path = os.path.join(directory_path, filename)
        # 读取文件内容
        with open(file_path, 'r') as file:
            # 将每行内容去除换行符后添加到列表中
            words.extend(line.strip() for line in file)

with open(detect_path, 'r') as file:
    prompts.extend(line.strip() for line in file)

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, text):
        for i in range(len(text)):
            node = self.root
            j = i
            while j < len(text) and text[j] in node.children:
                node = node.children[text[j]]
                if node.is_end_of_word:
                    return True
                j += 1
        return False

trie = Trie()
for word in words:
    trie.insert(word)

import datetime
from flask import Flask, request
import json


app = Flask(__name__)

@app.route('/')
def index():
    res={'msg':'这是一个接口','msg_dode':0}
    return json.dumps(res)

"""
    GET请求，带参数
"""
@app.route("/detect",methods=["POST"])
def detect():
    # 默认返回内容
    return_dict = {'return_info': 'success', 'result': 'safe'}

    data = request.get_json()
    user = data.get('user')
    assistant = data.get('assistant')

    # # 判断入参是否为空
    # if len(request.args) == 0:
    #     return_dict['return_info'] = 'failed - mou je'
    #     return_dict['result'] = None
    #     return json.dumps(return_dict, ensure_ascii=False)
    # # 获取传入的params参数
    # get_data = request.args.to_dict()

    # user = get_data.get('user')
    # assistant = get_data.get('assistant')
    
    if user is None:
        user = ''
    if assistant is None:
        assistant = ''

    # llama guard detect
    chat = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    llama_guard_response, user_num_tokens, assistant_num_tokens = moderate(chat)
    return_dict['user_num_tokens'] = user_num_tokens
    return_dict['assistant_num_tokens'] = assistant_num_tokens

    # prompt injection detection
    sanitized_prompt, is_valid, risk_score = scanner.scan(user)
    if not is_valid:
        return_dict['result'] = 'prompt injection detected!!!'
        return return_dict
    
    if 'unsafe' in llama_guard_response:
        return_dict['result'] = 'illegal content detected!!!'
        return return_dict
    
    # sensitive lexicon detection
    if trie.search(user) or trie.search(assistant):
        return_dict['result'] = 'sensitive lexicon detected!!!'
        return return_dict
    

    return return_dict

app.run(host="0.0.0.0", port=5000, debug=False)
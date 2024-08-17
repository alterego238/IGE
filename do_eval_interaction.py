import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from openai import OpenAI
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import  PeftModel
from gradio import ChatMessage
from colorama import Fore, Style, init
import threading
from tqdm import tqdm
import argparse
from utils import *

init(autoreset=True)
def print_user(message):
    print(Fore.BLUE + "User: " + message)
def print_assistant(message):
    print(Fore.GREEN + "Assistant: " + message)

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def init_model(load_model_path, cache_dir, load_ckpt, use_lora):
    print("init model ...")
    tokenizer = AutoTokenizer.from_pretrained(load_model_path, cache_dir=cache_dir)
    if use_lora:
        model = AutoModelForCausalLM.from_pretrained(load_model_path, cache_dir=cache_dir, device_map="auto")
        model = PeftModel.from_pretrained(model, load_ckpt, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(load_ckpt, cache_dir=cache_dir, device_map="auto")
    return model, tokenizer


def invoke_api(model, history):
    base_url, api_key = get_base_url_and_api_key()

    client = OpenAI(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model=model,
        messages=history
    )
    return completion.choices[0].message.content

def local_inference(model, tokenizer, history):
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n"
    
    input_ids = tokenizer.apply_chat_template(history, return_dict=True, add_generation_prompt=True)["input_ids"]
    input_ids = torch.LongTensor([input_ids])
    outputs = model.generate(input_ids=input_ids.to("cuda:0"),
                            max_new_tokens=700,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(outputs[0]))
    response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    return response

def inference(api, model, tokenizer, history):
    if api:
        return invoke_api(model, history)
    else:
        return local_inference(model, tokenizer, history)


def invoke_openai(model, history, index):
    base_url, api_key = get_base_url_and_api_key()
    api_key = get_api_key_for_index(index)

    client = OpenAI(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model=model,
        messages=history,
        temperature=0
    )
    return completion.choices[0].message.content

def inetract(api, user_model, model, tokenizer, user_system_prompt_path, assistant_system_prompt_path, script, worker_num):
    with open(user_system_prompt_path, 'r') as f_user, open(assistant_system_prompt_path, 'r') as f_assistant:
        user_system_prompt, assistant_system_prompt = f_user.read(), f_assistant.read()
    user_system_prompt = user_system_prompt + script
    user_history = [{"role": "system", "content": f"{user_system_prompt}"}]
    assistant_history = [{"role": "system", "content": f"{assistant_system_prompt}"}]
    
    '''assistant_history = [{"role": "user", "content": f"{assistant_system_prompt}"}, 
                            {"role": "assistant", "content": f"OK, I will be a assistant to help you develop a game."}]'''

    
    user_message = '[start interactive game development]'
    for i in range(20):
        assistant_history.append({"role": "user", "content": f"{user_message}"})
        
        assistant_message = inference(api, model, tokenizer, assistant_history)
        assistant_history.append({"role": "assistant", "content": f"{assistant_message}"})
        if worker_num == 0:
            print_user(user_message)
            print_assistant(assistant_message)
        if "script['Flow']" in assistant_message:
            break
        
        user_history.append({"role": "user", "content": f"{assistant_message}"})
        user_message = invoke_openai(user_model, user_history, worker_num)
        user_history.append({"role": "assistant", "content": f"{user_message}"})
        
    return assistant_history

def worker(worker_num, api, data_chunk, user_model, model, tokenizer, user_system_prompt_path, assistant_system_prompt_path):
    for d in tqdm(data_chunk, desc=f"Worker {worker_num}", position=worker_num):
        history = inetract(api, user_model, model, tokenizer, user_system_prompt_path, assistant_system_prompt_path, d['script'], worker_num)
        history.pop(0)
        d['history'] = history

def multithread(api, data, user_model, model, tokenizer, user_system_prompt_path, assistant_system_prompt_path, num_threads=None, chunk_size=None):
    n = len(data)
    if num_threads is not None:
        num_threads = min(num_threads, n)
        chunk_size = n // num_threads
    elif chunk_size is not None:
        num_threads = n // chunk_size + 1
    else:
        raise ValueError('num_threads or chunk_size must be specified')
    
    threads = []
    for i in range(num_threads):
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size if i != num_threads - 1 else n
        data_chunk = data[start_index:end_index]
        
        thread = threading.Thread(target=worker, args=(i, api, data_chunk, user_model, model, tokenizer, user_system_prompt_path, assistant_system_prompt_path))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

    print("All threads are finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model")
    parser.add_argument("--api", action="store_true", help="model")
    parser.add_argument("--no_ft", action="store_true", help="model")
    parser.add_argument("--wo_script", action="store_true", help="model")
    parser.add_argument("--wo_instruct", action="store_true", help="model")
    parser.add_argument("--wo_stage0", action="store_true", help="model")
    parser.add_argument("--num_threads", type=int, default=10, help="model")

    args = parser.parse_args()
    model = args.model
    api = args.api
    
    user_system_prompt_path = './prompt/interactor.md'
    assistant_system_prompt_path = './prompt/gpt_system_prompt.md' if api else './prompt/system_prompt.md'
    user_model = 'gpt-4o-mini'
    num_threads = args.num_threads
    chunk_size = None
    
    use_lora = True
    cache_dir = '../../cache'
    load_model_path = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    load_ckpt = f'./model/{model}'
    result_dir = f'./result/{model}'
    result_path = f'./result/{model}/test.jsonl'
    
    if args.no_ft:
        use_lora = False
        load_ckpt = load_model_path
        assistant_system_prompt_path = './prompt/gpt_system_prompt.md'
        
    if args.wo_script:
        assistant_system_prompt_path = './prompt/system_prompt_wo_script.md'
        user_system_prompt_path = './prompt/interactor_wo_script.md'
        
    if args.wo_instruct:
        load_model_path = 'meta-llama/Meta-Llama-3.1-8B'
        use_lora = False
        load_ckpt = load_model_path
        assistant_system_prompt_path = './prompt/gpt_system_prompt.md'
        
    if args.wo_stage0:
        load_model_path = 'meta-llama/Meta-Llama-3.1-8B'
    
    
    tokenizer = None
    if not api:
        model, tokenizer = init_model(load_model_path, cache_dir, load_ckpt, use_lora)
    
    data = []
    data_path = './data/test.jsonl'
    with open(data_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            data.append(d)

    create_dir(result_dir)
    multithread(api, data, user_model, model, tokenizer, user_system_prompt_path, assistant_system_prompt_path, num_threads=num_threads, chunk_size=chunk_size)
    
    with open(result_path, 'w') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
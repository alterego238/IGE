import os
import json
import yaml
import threading
from tqdm import tqdm
from openai import OpenAI
import re
import argparse
import textwrap
import itertools
import random
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from utils import *

def invoke_openai(model, prompt, index):
    api_key = get_api_key_for_index(index)
    base_url = "https://api2.aigcbest.top/v1"

    client = OpenAI(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": f"{prompt}"}
        ]
    )
    return completion.choices[0].message.content

def worker(worker_num, model, data_chunk):
    for d in tqdm(data_chunk, desc=f"Worker {worker_num}", position=worker_num):
        if 'prompt' in d.keys():
            prompt_input = d['prompt']
        else:
            raise ValueError('prompt must be in data')
        d['response'] = invoke_openai(model, prompt_input, worker_num)
        
def multithread_invoke_openai(model, data, num_threads=None, chunk_size=None):
    n = len(data)
    if n == 0:
        return 
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
        
        thread = threading.Thread(target=worker, args=(i, model, data_chunk))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

    print("All threads are finished.")
    
def convert_to_history(response):
    try:
        history = []
        pattern = r'(assistant|user):\s*\n(.*?)(?=\n(?:assistant|user):|\Z)'
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            role = match[0]
            content = match[1].strip()
            history.append({'role': role, 'content': content})
            
        return history
    except Exception as e:
        print(e)
        return []

def extract_code_script(text):
    text = text.strip()
    snippet_pattern = r'## new code snippet:\n(.*?)\n\n## new script segment:'
    script_segment_pattern = r'## new script segment:\n(.*)'

    code_snippet = re.search(snippet_pattern, text, re.DOTALL).group(1).strip()
    script_segment = re.search(script_segment_pattern, text, re.DOTALL).group(1).strip()
    return code_snippet, script_segment


def judge_code_exec(customgame_methods_path, code_template_path, code_snippet):
    with open(customgame_methods_path, 'r') as f_methods, open(code_template_path, 'r') as f_template:
        methods_str = f_methods.read().strip()
        code_template = f_template.read()
    methods_new = methods_str + '\n' + code_snippet
    methods_new = textwrap.indent(methods_new, '   ')
    code_complete = code_template.format(0, methods_new)
    
    generated_dict = {}
    try:
        exec(code_complete, generated_dict)
        return True
    except Exception as e:
        print(e)
        return False

def extract_tag_contents(text, tag):
    pattern = rf'<{tag}>(.*?)</{tag}>'
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) == 0:
        return ''
    return matches[0]


def filter_exec_code_script(script_code_pool_path, data_path, data_generated_history_path, data):
    with open(script_code_pool_path, 'a') as f_pool, open(data_path, 'w') as f_data, open(data_generated_history_path, 'a') as f_history:
        for line in data:
            try:
                line.pop('prompt')
                code, script = extract_code_script(line['response'])
                code = code.replace('```python', '').replace('```', '')
                code, script = code.strip(), script.strip()
                line.pop('response')
                line['edited_code'], line['edited_script'] = code, script
                f_data.write(json.dumps(line) + '\n')
                f_history.write(json.dumps(line) + '\n')
                if not judge_code_exec(customgame_methods_path, code_template_path, code):
                    continue
                f_pool.write(json.dumps({'script': script, 'code': code}) + '\n')
            except Exception as e:
                print(e)
                f_data.write(json.dumps(line) + '\n')
                f_history.write(json.dumps(line) + '\n')


def filter_exec_interaction(script_code_pool_path, completed_data, data):
    with open(script_code_pool_path, 'w') as f_pool:
        for line in completed_data:
            f_pool.write(json.dumps(line) + '\n')
        for line in data:
            history = convert_to_history(line['response'])
            if history:
                code_snippets = []
                try:
                    for message in history:
                        if message['role'] == 'assistant':
                            code_snippets.append(extract_tag_contents(message['content'], 'code').strip())
                    generated_code = '\n\n'.join(code_snippets)
                    if judge_code_exec(customgame_methods_path, code_template_path, generated_code):
                        line['history'] = history
                except Exception as e:
                    print(e)
            line.pop('response')
            line.pop('prompt')
            f_pool.write(json.dumps(line, ensure_ascii=False) + '\n')


def filter_interaction(script_code_pool_path, train_path):
    with open(script_code_pool_path, 'r') as f, open(train_path, 'w') as o:
        for line in f:
            data = json.loads(line)
            if 'history' not in data.keys() or 'assistant' not in [item['role'] for item in data['history']]:
                continue
            
            o.write(json.dumps(data) + '\n')

def evalueate_complete(d, code_template, script_code_pool_path):
    script_segments = []
    code_snippets = []
    
    if d['history'] == []:
        exec_success = False
        corrent = False
    else:
        for message in d['history']:
            if message['role'] == 'assistant':
                script_segments.append(extract_tag_contents(message['content'], 'script').strip())
                code_snippets.append(extract_tag_contents(message['content'], 'code').strip())
            
        generated_script = '\n'.join(script_segments)
        generated_code = '\n\n'.join(code_snippets)
        
        generated_code = textwrap.indent(generated_code, '    ')
        gt_code = textwrap.indent(d['code'], '    ')


        exec_success = True
        corrent = True
        seeds = random.sample(range(10000), 40)
        for seed in tqdm(seeds):
            try:
                generated_code_complete = code_template.format(seed, generated_code)
                gt_code_complete = code_template.format(seed, gt_code)
                gt_dict = {}
                exec(gt_code_complete, gt_dict)
                gt_state = gt_dict['game'].state
                
                generated_dict = {}
                exec(generated_code_complete, generated_dict)
                generated_state = generated_dict['game'].state
                
                if generated_state != gt_state:
                    corrent = False
            except Exception as e:
                print(e)
                exec_success = False
                corrent = False
                break
    
    with open(script_code_pool_path, 'a') as o:
        if not corrent:
            d.pop('history')
        o.write(json.dumps(d) + '\n')
    return exec_success, corrent

def multi_process_file(data_lst, code_template_path, script_code_pool_path):
    with open(code_template_path, "r") as f_code:
        code_template = f_code.read()

    process_line_with_params = partial(evalueate_complete, code_template=code_template, script_code_pool_path=script_code_pool_path)
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_line_with_params, data_lst)
    
    return results


def filter_exec_interaction_complete(script_code_pool_path, completed_data, data, code_template_path):
    with open(script_code_pool_path, 'w') as f_pool:
        for line in completed_data:
            f_pool.write(json.dumps(line) + '\n')
    for line in data:
        history = convert_to_history(line['response'])
        
        complete = False
        for item in history:
            if item['role'] == 'assistant':
                if "script['Flow']" in item['content']:
                    complete = True
                    break
        
        history.insert(0, {'role': 'user', 'content': '[start interactive game development]'})
        if complete:
            line['history'] = history
        else:
            line['history'] = []
        
        line.pop('response')
        line.pop('prompt')
            
    multi_process_file(data, code_template_path, script_code_pool_path)
    
def mix_data(train_path, train_complete_path, mixed_path):
    with open(mixed_path, 'w') as f_mix:
        with open(train_path, 'r') as f_train, open(train_complete_path, 'r') as f_train_complete:
            f_mix.write(f_train.read())
            f_mix.write(f_train_complete.read())
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, choices=['interaction', 'code_script', 'interaction_complete'], help="generate stage")
    
    args = parser.parse_args()

    demo = False
    model = 'gpt-4o'
    stage = args.stage
    num_threads = 10
    chunk_size = None

    prompt_path = f'./prompt/{stage}.md'
    data_path = './data/data.jsonl'
    data_generated_history_path = './data/data_generated_history.jsonl'
    
    script_code_pool_path = './data/script_code_pool.jsonl'
    train_path = './data/train.jsonl'
    complete_pool_path = './data/complete_pool.jsonl'
    train_complete_path = './data/train_complete.jsonl'
    mixed_path = './data/mixed.jsonl'
    
    customgame_methods_path = './CustomGame_methods.py'
    code_template_path = './CustomGame.py'

    if demo:
        with open(prompt_path, 'r') as f_prompt:
            prompt = f_prompt.read()

        prompt += '\n\n\n# Start of Official Requests\n'
        response = invoke_openai(model, prompt)
        response_path = prompt_path.replace('prompt', model + '_response')
        with open(response_path, 'w') as f_response:
            f_response.write(prompt + response)
    else:
        # preprocess
        with open(prompt_path, 'r') as f_prompt:
            prompt = f_prompt.read()
        #prompt += '# Start of Official Requests\n'
        
        data = []
        completed_data = []
        if stage == 'code_script':
            with open(data_path, 'r') as f_data:
                for line in f_data:
                    d = json.loads(line)
                    prompt_complete = prompt + f'# Start of Official Requests\n## original code snippet:\n{d["original_code"]}\n\n## original script segment:\n{d["original_script"]}\n\n'
                    d['prompt'] = prompt_complete
                    data.append(d)
        elif stage == 'interaction' or stage == 'interaction_complete':
            pool_path = script_code_pool_path if stage == 'interaction' else complete_pool_path
            with open(pool_path, 'r') as f_pool:
                for line in f_pool:
                    d = json.loads(line)
                    if 'history' in d.keys():
                        completed_data.append(d)
                        continue
                    code_snippet = textwrap.dedent('\n'.join(d['code'].split('\n')[1:]))
                    prompt_complete = prompt + f'# Start of Official Requests\n## script segment:\n{d["script"]}\n\n## code snippet:\n{d["code"]}\n\n## dialogue:\n'
                    d['prompt'] = prompt_complete
                    data.append(d)
        else:
            raise ValueError('stage must be one of [script, code, interaction]')


        # generate
        multithread_invoke_openai(model, data, num_threads=num_threads, chunk_size=chunk_size)

        
        # filter and save
        if stage == 'code_script':
            filter_exec_code_script(script_code_pool_path, data_path, data_generated_history_path, data)
        elif stage == 'interaction':
            filter_exec_interaction(script_code_pool_path, completed_data, data)
        elif stage == 'interaction_complete':
            filter_exec_interaction_complete(complete_pool_path, completed_data, data, code_template_path)
        else:
            raise ValueError('stage must be one of [script, code, interaction]')
        
        
        if stage == 'interaction':
            filter_interaction(script_code_pool_path, train_path)
        elif stage == 'interaction_complete':
            filter_interaction(complete_pool_path, train_complete_path)
            mix_data(train_path, train_complete_path, mixed_path)
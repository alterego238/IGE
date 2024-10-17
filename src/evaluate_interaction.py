import json
from openai import OpenAI
import threading
from tqdm import tqdm
import argparse
import re
import os
import queue
from utils import *
from collections import defaultdict

def invoke_openai(model, prompt, index):
    base_url, api_key = get_base_url_and_api_key()
    api_key = get_api_key_for_index(index)

    client = OpenAI(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": f"{prompt}"}
        ]
    )
    return completion.choices[0].message.content

def extract_tag_contents(text, tag):
    pattern = rf'<{tag}>(.*?)</{tag}>'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches == []:
        return ''
    return matches[0].strip()

def history_to_txt(history): 
    txt_lst = []
    for item in history:
        content = item['content']
        if item['role'] == 'assistant':
            content = extract_tag_contents(content, 'utter')
        message_txt = item['role'] + ':\n' + content
        txt_lst.append(message_txt)
    history_txt = '\n\n'.join(txt_lst)
    return history_txt

def worker(worker_num, data_chunk, evaluator_prompt_dir, model, result_queue, indexs):
    for dialogue in tqdm(data_chunk, desc=f"Worker {worker_num}", position=worker_num):
        result = {}
        if os.path.isdir(evaluator_prompt_dir):
            for file_name in os.listdir(evaluator_prompt_dir):
                with open(os.path.join(evaluator_prompt_dir, file_name), 'r') as f:
                    prompt = f.read()
                prompt = prompt.format(dialogue=dialogue)
                response = invoke_openai(model, prompt, worker_num)
                result[file_name.split('.')[0]] = response.strip()
        else:
            with open(evaluator_prompt_dir, 'r') as f:
                prompt = f.read()
            prompt = prompt.replace('{dialogue}', dialogue)
            while True:
                try:
                    response = invoke_openai(model, prompt, worker_num)
                    response = response.replace('```json', '').replace('```', '').strip()
                    result = json.loads(response)
                    break
                except Exception as e:
                    print(e)
                    print('retry')
                    
            for key, value in result.items():
                result[key] = value['score']
            
        result_queue.put((indexs.pop(0), result))

def multithread(data, evaluator_prompt_dir, model, num_threads=None, chunk_size=None):
    n = len(data)
    if num_threads is not None:
        num_threads = min(num_threads, n)
        chunk_size = n // num_threads
    elif chunk_size is not None:
        num_threads = n // chunk_size + 1
    else:
        raise ValueError('num_threads or chunk_size must be specified')
    
    threads = []
    result_queue = queue.Queue()
    
    for i in range(num_threads):
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size if i != num_threads - 1 else n
        data_chunk = data[start_index:end_index]

        thread = threading.Thread(target=worker, args=(i, data_chunk, evaluator_prompt_dir, model, result_queue, list(range(start_index, end_index))))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
        
    print("All threads are finished.")
    
    results = [result_queue.get() for _ in range(n)]
    results.sort()
    results = [result for _, result in results]
    
    return results

def evaluate_interaction(evaluator_model, test_data_path, evaluator_prompt_dir, num_threads):
    data = []
    with open(test_data_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            history_txt = history_to_txt(d['history'])
            data.append(history_txt)
            
    results = multithread(data, evaluator_prompt_dir, evaluator_model, num_threads)
    
    sums = defaultdict(int)
    counts = defaultdict(int)
    for item in results:
        for key, value in item.items():
            sums[key] += int(value)
            counts[key] += 1
    averages = {key: sums[key] / counts[key] for key in sums}
    
    return averages
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model")
    args = parser.parse_args()
    model = args.model
    
    evaluator_model = "gpt-4o"
    num_threads = 10
    
    test_data_path = f"../result/{model}/test.jsonl"
    result_path = f"../result/{model}/result.txt"
    
    #evaluator_prompt_path = "../prompt/evaluator.md"
    evaluator_prompt_dir = '../prompt/evaluator.md'
    if os.path.isdir(evaluator_prompt_dir) or '7' in evaluator_prompt_dir:
        score_scale = 7
    else:
        score_scale = 4
    
    averages_lst = []
    for _ in range(5):
        averages = evaluate_interaction(evaluator_model, test_data_path, evaluator_prompt_dir, num_threads)
        averages_lst.append(averages)
        
    sum_dict = {}
    for d in averages_lst:
        for key, value in d.items():
            sum_dict[key] = sum_dict.get(key, 0) + value
    count = len(averages_lst)
    avg_dict = {key: sum_dict[key] / count for key in sum_dict}
    
    avg_dict = {key: round(value / score_scale * 100, 2)for key, value in avg_dict.items()}
    
    with open(result_path, 'a') as f_result:
        f_result.write(json.dumps(avg_dict) + '\n')
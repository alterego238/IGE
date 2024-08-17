import os
import json
import yaml
import threading
from tqdm import tqdm
from openai import OpenAI
import re
from utils import *

def replace_newlines_in_quotes(s):
    def replace_newlines(match):
        return match.group(0).replace('\n', '\\n')
    pattern = r'\"(.*?)\"'
    result = re.sub(pattern, replace_newlines, s, flags=re.DOTALL)
    
    return result

def invoke_openai(model, prompt):
    base_url, api_key = get_base_url_and_api_key()

    client = OpenAI(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": f"{prompt}"}
        ]
    )
    return completion.choices[0].message.content

cnt = 0
def worker(worker_num, model, data_chunk, prompt):
    for d in tqdm(data_chunk, desc=f"Worker {worker_num}", position=worker_num):
        if 'prompt' in d.keys():
            prompt_input = d['prompt']
        else:
            user_input = [{'user': item['user']} for item in d['history']]
            inputs = '## input:\n### script:\n' + yaml.dump(d['script'], width=float('inf'), sort_keys=False) + '\n### seed data:\n' + json.dumps(user_input, indent=2) + '\n## output:\n'
            prompt_input = prompt + inputs
        d['response'] = invoke_openai(model, prompt_input)
        
        history = []
        response = d['response']
        str_historys = response.split('\n\nuser:')
        for i, s in enumerate(str_historys):
            if i == 0:
                user, assistant = s.split('\n\nassistant:')
                user_content, assistant_content = ':'.join((user.split(':')[1:])).strip(), assistant.strip()
                user_content, assistant_content = user_content.replace('\n\n', '\n'), assistant_content.replace('\n\n', '\n')
                history.append({'user': user_content, 'assistant': assistant_content})
            else:
                user, assistant = s.split('\n\nassistant:')
                user_content, assistant_content = user.strip(), assistant.strip()
                user_content, assistant_content = user_content.replace('\n\n', '\n'), assistant_content.replace('\n\n', '\n')
                history.append({'user': user_content, 'assistant': assistant_content})
            
        d['history'] = history

def transform(tmp_path, data_path):
    with open(tmp_path, 'r') as f, open(data_path, 'w') as f_train:
        cnt = 0
        for line in f:
            d = json.loads(line)
            try:
                response = json.loads(replace_newlines_in_quotes(d['response']).replace('```json', '')[:-3])
                history = response

                d_train = {'prompt': d['prompt'], 'history': history}
                json.dump(d_train, f_train)
                f_train.write('\n')
            except Exception as e:
                print(d['response'].replace('```json', '')[:-3])
                print(e)
                cnt += 1
        print('error:', cnt)

if __name__ == '__main__':
    demo = True
    model = 'gpt-4o'
    stage = 'interaction'

    input_path = './data/seed_data.jsonl'
    tmp_path = f'./data/tmp_{stage}.jsonl'
    output_path = f'./data/{stage}_no_scene.jsonl'

    prompt_path = f'./{stage}/prompt_only_utter.md'
    examples_dir = f'./{stage}/examples'
    response_dir = f'./{stage}/response'

    #end = '_round'
    if demo:
        input_dir = f'./{stage}/input'

        with open(prompt_path, 'r') as f_prompt:
            prompt = f_prompt.read()
            
        if os.path.exists(examples_dir):
            for i, example_filename in enumerate(sorted(os.listdir(examples_dir))):
                example_path = os.path.join(examples_dir, example_filename)
                with open(example_path, 'r') as f_example:
                    prompt += f'\n\n\n# Example {i + 1}\n' + f_example.read()

            prompt += '\n\n\n# Start of Official Requests\n'

        if os.path.exists(input_dir):
            for input_filename in sorted(os.listdir(input_dir)):
                input_path = os.path.join(input_dir, input_filename)
                with open(input_path, 'r') as f_input:
                    prompt_input = prompt + f_input.read()
                response = invoke_openai(model, prompt_input)

                response_path = os.path.join(response_dir, model + '_' + input_path.split('/')[-1])
                with open(response_path, 'w') as f_response:
                    f_response.write(prompt_input + response)
        else:
            response = invoke_openai(model, prompt)
            
            if os.path.exists(response_dir):
                response_path = os.path.join(response_dir, model + '_' + input_path.split('/')[-1])
            else:
                response_path = prompt_path.replace('prompt', model + '_response')
            with open(response_path, 'w') as f_response:
                f_response.write(prompt + response)
    else:
        data = []
        with open(input_path, 'r') as f_input:
            for line in f_input:
                data.append(json.loads(line))

        if 'test' in output_path:
            data = data[:4]
            num_threads = 2
        else:
            num_threads = 10

        '''with open(prompt_path, 'r') as f_prompt:
            prompt = f_prompt.read()'''
        """for i, example_filename in enumerate(sorted(os.listdir(examples_dir))):
            example_path = os.path.join(examples_dir, example_filename)
            with open(example_path, 'r') as f_example:
                prompt += f'\n\n\n# Example {i + 1}\n' + f_example.read()

        prompt += '\n\n\n# Start of Official Requests\n'"""

        n = len(data)
        
        chunk_size = n // num_threads
        
        threads = []
        for i in range(num_threads):
            start_index = i * chunk_size
            end_index = (i + 1) * chunk_size if i != num_threads - 1 else n
            data_chunk = data[start_index:end_index]
            
            thread = threading.Thread(target=worker, args=(i, model, data_chunk, ''))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()

        print("All threads are finished.")

        with open(output_path, 'w') as f_output:
            for line in data:
                #line.pop('response')
                json.dump(line, f_output, ensure_ascii=False)
                f_output.write('\n')
        
        #transform(tmp_path, output_path)
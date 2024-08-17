import json
import yaml
import re
import textwrap
import os
import random
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import shutil
import argparse
from openai import OpenAI
import concurrent.futures

def clear_directory(directory_path):
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if filename == 'GameBase.py':
                continue
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'删除 {file_path} 失败. 理由: {e}')
    else:
        print(f'目录 {directory_path} 不存在或不是一个目录')
        
def extract_tag_contents(text, tag):
    pattern = rf'<{tag}>(.*?)</{tag}>'
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) == 0:
        return ''
    return matches[0]

def judge_code_exec(code_template, gt_code_methods_indent, generated_code_methods_indent):
    exec_success = True
    corrent = True
    seeds = random.sample(range(10000), 40)
    for seed in tqdm(seeds):
        try:
            generated_code_complete = code_template.format(seed, generated_code_methods_indent)
            gt_code_complete = code_template.format(seed, gt_code_methods_indent)
            gt_dict = {}
            exec(gt_code_complete, gt_dict)
            gt_state = gt_dict['game'].state
            
            generated_dict = {}
            exec(generated_code_complete, generated_dict)
            generated_state = generated_dict['game'].state
            
            if generated_state != gt_state:
                corrent = False
                break
        except Exception as e:
            print(e)
            exec_success = False
            corrent = False
            break
    return exec_success, corrent, generated_code_complete, seed

def evalueate_complete(i_line, code_template, error_data_path, error_dir=None):
    i, line = i_line
    d = json.loads(line)
    if 'history' not in d.keys() or not isinstance(d['history'], list):
        return False, False
    
    script_segments = []
    code_snippets = []

    gt_code_methods = d['code']
    gt_code_methods_indent = textwrap.indent(gt_code_methods, '    ')
    
    config_complete = False
    num_functions = 0
    func_exec_success, func_corrent = 0, 0
    static_methods = ['config', 'start', 'shuffle', 'switch', 'dealx', 'flopx', 'bet_done', 'blind', 'set_flow']
    static_methods_dict = {method: {'num_funcitons': 0, 'func_exec_success': 0, 'func_correct': 0} for method in static_methods}
    for message in d['history']:
        if message['role'] == 'assistant':
            #script_segment = extract_tag_contents(message['content'], 'script').strip()
            code_snippet = extract_tag_contents(message['content'], 'code').strip()
            #script_segments.append(script_segment)
            code_snippets.append(code_snippet)
            
            generated_code_methods = gt_code_methods + '\n\n' + code_snippet
            generated_code_methods_indent = textwrap.indent(generated_code_methods, '    ')
            
            exec_success, corrent, generated_code_complete, seed = judge_code_exec(code_template, gt_code_methods_indent, generated_code_methods_indent)
            
            if 'config(self)' in code_snippet and not exec_success:
                continue
            
            matching_substrings = [sub for sub in static_methods if 'def ' + sub in code_snippet]
            for method in matching_substrings:
                static_methods_dict[method]['num_funcitons'] += 1
                static_methods_dict[method]['func_exec_success'] += exec_success
                static_methods_dict[method]['func_correct'] += corrent
            
            num_functions += 1
            func_exec_success += exec_success
            func_corrent += corrent
            
            if not corrent:
                with open(error_data_path, 'a') as f_err_data:
                    err_dict = {'i': i, 'seed': seed, 'script': d['script'], 'exec_success': exec_success, 'corrent': corrent, 'code_snipet': code_snippet}
                    f_err_data.write(json.dumps(err_dict) + '\n')
                        
                        
    if num_functions == 0:
        return False, False, 0, 0, 0, static_methods_dict
    
    func_exec_success_rate = func_exec_success / num_functions
    func_corrent_rate = func_corrent / num_functions

    generated_script = '\n'.join(script_segments)
    generated_code_methods = '\n\n'.join(code_snippets)
    generated_code_methods_indent = textwrap.indent(generated_code_methods, '    ')
    
    exec_success, corrent, generated_code_complete, seed = judge_code_exec(code_template, gt_code_methods_indent, generated_code_methods_indent)
    
    if not corrent:
        with open(os.path.join(error_dir, f'generated_code_{i}.py'), "w") as f_err:
            f_err.write(generated_code_complete)
                
    return exec_success, corrent, func_exec_success_rate, func_corrent_rate, num_functions, static_methods_dict


def multi_process_file(file_path, code_template_path, error_data_path, error_dir):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    with open(code_template_path, "r") as f_code:
        code_template = f_code.read()

    i_lines = [(i, line) for i, line in enumerate(lines)]
    process_line_with_params = partial(evalueate_complete, code_template=code_template, error_data_path=error_data_path, error_dir=error_dir)
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_line_with_params, i_lines)
    
    return results

def evaluate_acc(test_data_path, code_template_path, result_path, error_data_path, error_dir=None, num_processes=1):
    with open(test_data_path, 'r') as f:
        n = sum(1 for line in f if 'history' in json.loads(line))
    with open(code_template_path, "r") as f_code:
        code_template = f_code.read()
    
    results = multi_process_file(test_data_path, code_template_path, error_data_path, error_dir)

    exec_success_rate = sum([1 for exec_success, _, _, _, _, _ in results if exec_success]) / n
    correct_rate = sum([1 for _, correct, _, _, _, _ in results if correct]) / n
    func_exec_success_rate = sum([fesr for _, _, fesr, _, _, _ in results]) / n
    func_corrent_rate = sum([fcr for _, _, _, fcr, _, _ in results]) / n
    num_functions = sum([nfc for _, _, _, _, nfc, _ in results])
    static_methods_dict_lst = [smd for _, _, _, _, _, smd in results]
    
    static_methods = ['config', 'start', 'shuffle', 'switch', 'dealx', 'flopx', 'bet_done', 'blind', 'set_flow']
    combined_static_methods_dict = {method: {'num_funcitons': 0, 'func_exec_success': 0, 'func_correct': 0} for method in static_methods}

    # 遍历列表中的每个字典，累加每个方法的值
    for d in static_methods_dict_lst:
        for method, stats in d.items():
            combined_static_methods_dict[method]['num_funcitons'] += stats['num_funcitons']
            combined_static_methods_dict[method]['func_exec_success'] += stats['func_exec_success']
            combined_static_methods_dict[method]['func_correct'] += stats['func_correct']
            
    for method, stats in combined_static_methods_dict.items():
        num_functions = stats['num_funcitons']
        if num_functions > 0:
            stats['func_exec_success_ratio'] = stats['func_exec_success'] / num_functions
            stats['func_correct_ratio'] = stats['func_correct'] / num_functions
        else:
            stats['func_exec_success_ratio'] = 0  # 或者你可以设为0或其他值，根据你的需求
            stats['func_correct_ratio'] = 0  # 或者你可以设为0或其他值，根据你的需求
    
    combined_static_methods_dict_str = json.dumps(combined_static_methods_dict, indent=2)
            
    print(f'exec_success_rate: {exec_success_rate: .4f}')
    print(f'correct_rate: {correct_rate: .4f}')
    print(f'function_exec_success_rate: {func_exec_success_rate: .4f}')
    print(f'function_correct_rate: {func_corrent_rate: .4f}')
    print(f'num_functions: {num_functions}')
    print(f'combined_static_methods_dict:\n{combined_static_methods_dict_str}')
    
    
    with open(result_path, "a") as f_result:
        f_result.write(f'exec_success_rate: {exec_success_rate: .4f}\n')
        f_result.write(f'correct_rate: {correct_rate: .4f}\n')
        f_result.write(f'function_exec_success_rate: {func_exec_success_rate: .4f}\n')
        f_result.write(f'function_correct_rate: {func_corrent_rate: .4f}\n')
        f_result.write(f'num_functions: {num_functions}\n')
        f_result.write(f'combined_static_methods_dict:\n{combined_static_methods_dict_str}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model")
    args = parser.parse_args()
    model = args.model
    
    test_data_path = f"./result/{model}/test.jsonl"
    error_data_path = f"./result/{model}/error.jsonl"
    code_template_path = "./CustomGame.py"
    error_dir = "./error_interaction"
    result_path = f"./result/{model}/result.txt"
    
    clear_directory(error_dir)
    if os.path.exists(error_data_path):
        os.remove(error_data_path)
    evaluate_acc(test_data_path, code_template_path, result_path, error_data_path, error_dir)
    
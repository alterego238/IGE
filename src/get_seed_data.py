import json
import yaml
import os
import re
import random
import textwrap
import ast
import argparse

def extract_functions(code):
    tree = ast.parse(code)
    functions = {}
    
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            func_body = ast.get_source_segment(code, node)
            functions[func_name] = func_body
            
    return functions

def add_pool(script_path, script_code_pool_path):
    with open(script_code_pool_path, 'a') as f_pool:
        with open(script_path, 'r') as f_script:
            for line in f_script:
                d = json.loads(line)
                script = d['script']
                code = d['code']
                
                config, phases, flow = script.split('\n\n')
                config, phases, flow = config.strip(), phases.strip(), flow.strip()

                functions = extract_functions(code)

                data = {'script': config, 'code': functions['config']}
                f_pool.write(json.dumps(data) + '\n')
                
                fun_dict = {
                    'start': 'start', 
                    'shuffle': 'shuffle', 
                    'blind': 'blind', 
                    'dealx': 'dealx', 
                    'flopx': 'flopx', 
                    'switch': 'switch', 
                    'bet': 'bet_done'
                }
                """fun_dict = {
                    'blind': 'blind', 
                    'dealx': 'dealx', 
                    'flopx': 'flopx', 
                }"""
                for line in phases.split('\n')[1:]:
                    phase_name = line.split(':')[0].strip()
                    if phase_name not in fun_dict:
                        continue
                    data = {
                        'script': 'Phase:\n' + line,
                        'code': functions[fun_dict[phase_name]]
                    }
                    f_pool.write(json.dumps(data) + '\n')
                
                data = {
                    'script': flow,
                    'code': functions['set_flow']
                }
                f_pool.write(json.dumps(data) + '\n')

def generate_seed_data(script_code_pool_path, data_path, sample_rate=1, inject_rate=1):
    with open(script_code_pool_path, 'r') as f_pool, open(data_path, 'w') as f_data:
        for line in f_pool:
            rnd = random.random()
            if rnd >= sample_rate:
                continue
            data = json.loads(line)
            script = data['script']
            code = data['code']
            '''if 'Config:' not in script and 'blind:' not in script and 'dealx:' not in script and 'flopx:' not in script:
                continue'''
            data = {
                'original_script': script,
                'original_code': code
            }
            f_data.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, choices=['add_pool', 'seed_data'], help="generate stage")
    args = parser.parse_args()
    
    script_path = '../data/complete_pool.jsonl'
    script_code_pool_path = '../data/script_code_pool.jsonl'
    data_path = '../data/data.jsonl'
    sample_rate = 0.8
    inject_rate = 0.5
    
    if args.stage == 'add_pool':
        add_pool(script_path, script_code_pool_path)
    elif args.stage == 'seed_data':
        generate_seed_data(script_code_pool_path, data_path, sample_rate=sample_rate, inject_rate=inject_rate)
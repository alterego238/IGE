import random
import re
import os
import json
import argparse
import textwrap

def judge_code_exec(code_run_path, code):
    with open(code_run_path, 'r') as f:
        code_run = f.read()
    code = textwrap.indent(code, '   ')
    code_complete = code_run.format(0, code)
    
    generated_dict = {}
    try:
        exec(code_complete, generated_dict)
        return True
    except Exception as e:
        print(e)
        return False

def replace_config_refs(text, config):
    pattern = r"config\['(.*?)'\]"
    
    def replace_func(match):
        key = match.group(1)
        return str(config.get(key, f"config['{key}']"))
    
    result = re.sub(pattern, replace_func, text)
    
    return result


def generate_seed_data(script_template_path, code_template_path, code_run_path, method_dir, data_path, num_samples, valid_dict):
    with open(script_template_path, 'r') as f_script_template, open(code_template_path, 'r') as f_code_template:
        script_template = f_script_template.read().strip()
        code_template = f_code_template.read().strip()
    
    
    blind_lst = []
    deal_lst = []
    flop_lst = []
    for sub_dir in os.listdir(method_dir):
        if sub_dir in valid_dict:
            for file_name in os.listdir(os.path.join(method_dir, sub_dir)):
                if file_name in valid_dict[sub_dir]:
                    with open(os.path.join(method_dir, sub_dir, file_name), 'r') as f_method:
                        method_code = f_method.read().strip()
                        desc, code = method_code.split('<SEP>')
                        eval(sub_dir + '_lst').append({'tag': file_name, 'desc': desc.strip(), 'code': code.strip()})

    card_combinations_rank = ['High Card', 'Pair', 'Two Pair', 'Three of a Kind', 'Straight', 'Flush', 'Full House', 'Four of a Kind', 'Straight Flush']
    
    with open(data_path, 'w') as f_pool:
        i = 0
        while i < num_samples:
            random.shuffle(card_combinations_rank)
            blind = random.choice(blind_lst)
            deal = random.choice(deal_lst)
            flop = random.choice(flop_lst)
            k = random.randint(3, 8)
            
            v_num = random.randint(10, 20)
            value = random.sample(range(0, 20), v_num)
            random.shuffle(value)
            
            config = {
                "n_players": random.randint(3, 6),
                "min_bet": random.choice([2, 4, 6, 8, 10]),
                "max_bet": random.choice([500, 800, 1000, 2000, 5000, 10000]),
                "suit": random.sample(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), k=k),
                "suit_have_rank": random.choice([True, False]),
                "value": random.choice([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1], [1, 11, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14], [7, 3, 2, 20, 18, 11, 6, 8, 4, 17, 12, 9, 5], value]),
                "card_combinations_rank": card_combinations_rank,
                "flow": random.choice([
                    ["start", "shuffle", "blind", "deal2", "bet", "flop3", "bet", "flop1", "bet", "flop1", "bet", "show", "prize"],
                    ["start", "shuffle", "deal2", "bet", "flop3", "bet", "flop1", "bet", "flop1", "bet", "show", "prize"],
                    ["start", "shuffle", "blind", "deal4", "bet", "flop3", "bet", "flop1", "bet", "flop1", "bet", "show", "prize"],
                    ["start", "shuffle", "blind", "deal3", "bet", "flop3", "bet", "flop1", "show", "prize"],
                    ["start", "shuffle", "deal3", "bet", "flop3", "bet", "flop1", "bet", "flop1", "bet", "show", "prize"],
                    ["start", "shuffle", "blind", "deal2", "bet", "flop3", "deal1", "bet", "flop1", "deal1", "bet", "show", "prize"]
                ]),
                "blind_code": blind['code'],
                "dealx_code": deal['code'],
                "flopx_code": flop['code'],
                "blind": blind['desc'],
                "dealx": deal['desc'],
                "flopx": flop['desc']
            }
            
            script = script_template.format_map(config)
            code = replace_config_refs(code_template, config)
            if not judge_code_exec(code_run_path, code):
                continue
            
            """with open('script_example.yaml', 'w') as f_script, open('code_example.py', 'w') as f_code:
                f_script.write(script)
                f_code.write(code)
            exit(1)"""
                
            data = {
                'script': script,
                'code': code
            }
            f_pool.write(json.dumps(data) + '\n')

            i += 1
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=['train', 'test', 'train_complete'], help="generate stage")
    args = parser.parse_args()
    
    script_template_path = './script_template.yaml'
    code_template_path = './CustomGame_template.py'
    code_run_path = './CustomGame.py'
    method_dir = '../fake'
    split = args.split
    
    tough = False
    
    if split == 'test':
        num_samples = 10
        data_path = '../data/test.jsonl'
        valid_dict = {
            'blind': ['lzs.txt', 'wkd.txt'], # 'gdk.txt'
            'deal': ['phr.txt', 'efl.txt', 'skz.txt', 'ajn.txt'], # 'tna.txt'
            'flop': ['imm.txt', 'cgw.txt', 'myy.txt'], # 'yje.txt', 'jti.txt
        }
        if tough:
            valid_dict['blind'].append('gdk.txt')
    elif split == 'train':
        num_samples = 20
        data_path = '../data/complete_pool.jsonl'
        valid_dict = {
            'blind': ['standard.txt', 'tua.txt', 'kea.txt'],
            'deal': ['standard.txt', 'ccz.txt', 'eqk.txt', 'hde.txt', 'hkx.txt', 'jeh.txt', 'jqw.txt', 'ppp.txt', 'qzi.txt', 'bte.txt'],
            'flop': ['standard.txt', 'apt.txt', 'buv.txt', 'jti.txt', 'kks.txt', 'nsf.txt', 'obn.txt', 'wyp.txt', 'yje.txt'],
        }
    elif split == 'train_complete':
        num_samples = 100
        data_path = '../data/complete_pool.jsonl'
        valid_dict = {
            'blind': ['standard.txt', 'tua.txt', 'kea.txt'],
            'deal': ['standard.txt', 'ccz.txt', 'eqk.txt', 'hde.txt', 'hkx.txt', 'jeh.txt', 'jqw.txt', 'ppp.txt', 'qzi.txt', 'bte.txt'],
            'flop': ['standard.txt', 'apt.txt', 'buv.txt', 'jti.txt', 'kks.txt', 'nsf.txt', 'obn.txt', 'wyp.txt', 'yje.txt'],
        }
    generate_seed_data(script_template_path, code_template_path, code_run_path, method_dir, data_path, num_samples, valid_dict)
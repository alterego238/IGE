import json
import re

with open("./data/train.jsonl", "r") as f:
    data = [json.loads(line) for line in f]
    
with open("./data/train_complete.jsonl", "r") as f:
    data_complete = [json.loads(line) for line in f]

with open("./data/mixed.jsonl", "w") as f:
    for line in data:
        f.write(json.dumps(line) + "\n")
    for line in data_complete:
        f.write(json.dumps(line) + "\n")
        

def remove_tagged_content(text, tag):
    pattern = f'<{tag}[^>]*>.*?</{tag}>'
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text

    
with open("./data/train_wo_script.jsonl", "w") as f:
    for d in data:
        for message in d['history']:
            if message['role'] == 'assistant':
                message['content'] = remove_tagged_content(message['content'], 'script').strip()
        f.write(json.dumps(d) + "\n")
        
with open("./data/train_complete_wo_script.jsonl", "w") as f:
    for d in data_complete:
        for message in d['history']:
            if message['role'] == 'assistant':
                message['content'] = remove_tagged_content(message['content'], 'script').strip()
        f.write(json.dumps(d) + "\n")
import argparse
import json
import logging
import os
import math
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import Trainer, TrainingArguments, set_seed
from accelerate.logging import get_logger
from accelerate import Accelerator
import evaluate

import re
import copy
import numpy as np
from datasets import load_dataset
from functools import partial


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = get_logger(__name__)


def get_input_ids(prompt_response, prompt, tokenizer):
    '''if tokenizer.chat_template is None:
        full_tokenized_ids = [tokenizer.bos_token_id] + tokenizer(prompt_response, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        prompt_input_ids = [tokenizer.bos_token_id] + tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_input_ids = full_tokenized_ids[len(prompt_input_ids):]
    else:'''
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n"
        
    full_tokenized_ids = tokenizer.apply_chat_template(prompt_response, return_dict=True)["input_ids"]
    prompt_input_ids = tokenizer.apply_chat_template(prompt, return_dict=True, add_generation_prompt=True)["input_ids"]
    answer_input_ids = full_tokenized_ids[len(prompt_input_ids):]

    # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
    full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])
    full_input_ids = np.array(full_tokenized_ids)
    assert len(full_input_ids) == len(full_concat_input_ids)

    # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
    # can be merged together when tokenizing prompt+answer. This could result
    # on the last token from the prompt being different when tokenized on its own
    # vs when done as prompt+answer.
    response_token_ids_start_idx = len(prompt_input_ids)

    # If tokenized prompt is different than both prompt+answer, then it means the
    # last token has changed due to merging.
    if prompt_input_ids != full_tokenized_ids[:response_token_ids_start_idx]:
        response_token_ids_start_idx -= 1

    full_input_ids = full_tokenized_ids
    prompt_input_ids = full_tokenized_ids[:response_token_ids_start_idx]
    answer_input_ids = full_tokenized_ids[response_token_ids_start_idx:]
    return full_input_ids, prompt_input_ids, answer_input_ids


def merge_labels(short_array, long_array):
    short_array = np.array(short_array)
    long_array = np.array(long_array)
    new_array = long_array.copy()
    mask = short_array != -100
    new_array[:len(short_array)][mask] = short_array[mask]
    return new_array.tolist()

def tokenize_row(example, tokenizer, max_seq_length, set_type, system_prompt_path, find_max_length=False):
    if 'history' in example.keys():
        with open(system_prompt_path, 'r') as f:
            system_prompt = f.read()
            
        '''history= [
            {"role": "system", "content": f"{system_prompt}"},
        ]
        for item in example['history']:
            history.append({"role": "user", "content": f"{item['user']}"})
            history.append({"role": "assistant", "content": f"{item['assistant']}"})'''
            
        history = example['history']
        history.insert(0, {"role": "system", "content": f"{system_prompt}"})

        labels = []
        prompt_response = []
        prompt = []
        for item in history:
            if item['role'] == 'system' or item['role'] == 'user':
                prompt_response.append(item)
                prompt.append(item)
            elif item['role'] == 'assistant':
                prompt_response.append(item)
                full_input_ids, prompt_input_ids, answer_input_ids = get_input_ids(prompt_response, prompt, tokenizer)
                input_ids = full_input_ids
                cur_labels = [-100] * (len(input_ids) - len(answer_input_ids)) + answer_input_ids
                labels = merge_labels(labels, cur_labels)
                prompt.append(item)
        if find_max_length:
            return {'max_total_len': [len(full_input_ids)], 'max_prompt_len': [len(prompt_input_ids)], 'max_response_len': [len(answer_input_ids)]}
        if set_type == 'test':
            input_ids = prompt_input_ids
            labels = answer_input_ids
        attention_mask = [1] * len(input_ids)

        offset_length = max_seq_length - len(input_ids)
        if offset_length >= 0:
            input_ids = [tokenizer.pad_token_id] * offset_length + input_ids
            attention_mask = [0] * offset_length + attention_mask
            if set_type == "test":
                labels = [-100] * (max_seq_length - len(labels)) + labels
            else:
                labels = [-100] * offset_length + labels
        else:
            input_ids = input_ids[-offset_length:]
            attention_mask = attention_mask[-offset_length:]
            if set_type == "test":
                labels = [-100] * (max_seq_length - len(labels)) + labels
            else:
                labels = labels[-offset_length:]

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(labels) == max_seq_length

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    

    if tokenizer.chat_template is None:
        prompt_response = "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n{response}".format_map(example)
        prompt = "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".format_map(example)
    else:
        system_prompt = 'Be a game engine for "Werewolf".\nAccording to the following game script and the current game state, output the the next game state change relative to the current game state.'
        instruction, inputs, response = example['instruction'], example['input'], example['response']

        prompt_response = [
            {"role": "system", "content": f"{system_prompt}"},
            {"role": "user", "content": f"Game script:\n{instruction}\n\nCurrent game state:\n{inputs}\n\nNext game state change:\n"},
            {"role": "assistant", "content": f"{response}"}
        ]
        prompt = [
            {"role": "system", "content": f"{system_prompt}"},
            {"role": "user", "content": f"Game script:\n{instruction}\n\nCurrent game state:\n{inputs}\n\nNext game state change:\n"}
        ]

    full_input_ids, prompt_input_ids, answer_input_ids = get_input_ids(prompt_response, prompt, tokenizer)

    if find_max_length:
        return {'max_total_len': [len(full_input_ids)], 'max_prompt_len': [len(prompt_input_ids)], 'max_response_len': [len(answer_input_ids)]}
    else:
        if set_type == 'test':
            input_ids = prompt_input_ids
            labels = answer_input_ids
        else:
            input_ids = full_input_ids
            labels = [-100] * (len(input_ids) - len(answer_input_ids)) + answer_input_ids
        attention_mask = [1] * len(input_ids)

        offset_length = max_seq_length - len(input_ids)
        if offset_length >= 0:
            input_ids = [tokenizer.pad_token_id] * offset_length + input_ids
            attention_mask = [0] * offset_length + attention_mask
            if set_type == "test":
                labels = [-100] * (max_seq_length - len(labels)) + labels
            else:
                labels = [-100] * offset_length + labels
        else:
            input_ids = input_ids[-offset_length:]
            attention_mask = attention_mask[-offset_length:]
            if set_type == "test":
                labels = [-100] * (max_seq_length - len(labels)) + labels
            else:
                labels = labels[-offset_length:]

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(labels) == max_seq_length

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def main():
    parser = argparse.ArgumentParser()

    # Data config
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Directory to contain the input data for all tasks.")
    parser.add_argument("--load_model_path", type=str, default="codellama/CodeLlama-7b-Instruct-hf",
                        help="Pre-trained language model to load.")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory to store the pre-trained language models downloaded from s3.")
    parser.add_argument("--output_dir", type=str, default="model/",
                        help="Directory to output predictions and checkpoints.")
    parser.add_argument("--load_ckpt", type=str, default="",
                        help="Checkpoint to load for trianing or evaluation.")

    # Training config
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to evaluate on the dev set.")
    parser.add_argument("--do_test", action="store_true",
                        help="Whether to evaluate on the test set.")
    parser.add_argument("--do_test_round", action="store_true",
                        help="Whether to evaluate on the test set.")
    parser.add_argument("--train_on", type=str, default="",
                        help="Choose a training set.")
    parser.add_argument("--eval_on", type=str, default="",
                        help="Choose a dev set.")
    parser.add_argument("--test_on", type=str, default="",
                        help="Choose a test set.")
    parser.add_argument("--results", type=str, default="results.json",
                        help="Choose a test set.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum total input sequence length after word-piece tokenization.")
    parser.add_argument("--train_batch_size", type=int, default=128,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=256,
                        help="Total batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Peak learning rate for optimization.")
    parser.add_argument("--num_train_epochs", type=float, default=3.0,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform (overrides training epochs).")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="Scheduler type for learning rate warmup.")
    parser.add_argument("--warmup_steps", type=float, default=30,
                        help="Proportion of training to perform learning rate warmup for.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="L2 weight decay for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward pass.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use mixed precision.")
    parser.add_argument("--bf16", action="store_true",
                        help="Whether to use mixed precision.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization.")
    parser.add_argument("--lora", action="store_true",
                        help="Whether to use low rank adaption.")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Whether to use DeepSpeed.")
    parser.add_argument("--adv_norm", type=float, default=0.,
                        help="Impose noise to input embeddings.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Impose noise to input embeddings.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Whether not to use CUDA when available.")
    parser.add_argument("--use_flash_attention_2", action="store_true",
                        help="Whether not to use CUDA when available.")
    parser.add_argument("--find_max_length", action="store_true",
                        help="Whether not to use CUDA when available.")
    parser.add_argument("--map_num_proc", type=int, default=4,
                        help="Random seed for initialization.")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Random seed for initialization.")
    parser.add_argument("--system_prompt_path", type=str, default='./system_prompt.txt',
                        help="Random seed for initialization.")
    
    args = parser.parse_args()

    set_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.load_model_path,
                                              do_lower_case=args.do_lower_case,
                                              padding_side='left',
                                              cache_dir=args.cache_dir)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def get_remove_columns(path):
        with open(os.path.join(path), 'r') as f:
            remove_columns = list(json.loads(next(f)).keys())
        return remove_columns

    # find_max_length
    if args.find_max_length:
        tokenize_row_train = partial(tokenize_row, tokenizer=tokenizer, max_seq_length=args.max_seq_length, set_type='train', system_prompt_path=args.system_prompt_path, find_max_length=True)
        tokenize_row_test = partial(tokenize_row, tokenizer=tokenizer, max_seq_length=args.max_seq_length, set_type='test', system_prompt_path=args.system_prompt_path, find_max_length=True)
        if args.do_train:
            remove_columns = get_remove_columns(os.path.join(args.data_dir, args.train_on))
            train_dataset = load_dataset("json", data_files=os.path.join(args.data_dir, args.train_on), split='train')
            eval_dataset = load_dataset("json", data_files=os.path.join(args.data_dir, args.eval_on), split='train') if args.do_eval else None
            train_max_lengths = train_dataset.map(tokenize_row_train, remove_columns=remove_columns, num_proc=args.map_num_proc)
            eval_max_lengths = eval_dataset.map(tokenize_row_train, remove_columns=remove_columns, num_proc=args.map_num_proc) if args.do_eval else None
            max_total_len = max(train_max_lengths['max_total_len'])
            max_prompt_len = max(train_max_lengths['max_prompt_len'])
            max_response_len = max(train_max_lengths['max_response_len'])
            if args.do_eval:
                max_total_len = max(max(eval_max_lengths['max_total_len']), max_total_len)
                max_prompt_len = max(max(eval_max_lengths['max_prompt_len']), max_prompt_len)
                max_response_len = max(max(eval_max_lengths['max_response_len']), max_response_len)
            print(f'max_total_len: {max_total_len}')
            print(f'max_prompt_len: {max_prompt_len}')
            print(f'max_response_len: {max_response_len}')
        if args.do_test:
            remove_columns = get_remove_columns(os.path.join(args.data_dir, args.test_on))
            test_dataset = load_dataset("json", data_files=os.path.join(args.data_dir, args.test_on), split='train')
            test_max_lengths = test_dataset.map(tokenize_row_test, remove_columns=remove_columns, num_proc=args.map_num_proc)
            max_total_len = max(test_max_lengths['max_total_len'])
            max_prompt_len = max(test_max_lengths['max_prompt_len'])
            max_response_len = max(test_max_lengths['max_response_len'])
            print(f'max_total_len: {max_total_len}')
            print(f'max_prompt_len: {max_prompt_len}')
            print(f'max_response_len: {max_response_len}')

    
    tokenize_row_train = partial(tokenize_row, tokenizer=tokenizer, max_seq_length=args.max_seq_length, set_type='train', system_prompt_path=args.system_prompt_path)
    tokenize_row_test = partial(tokenize_row, tokenizer=tokenizer, max_seq_length=args.max_seq_length, set_type='test', system_prompt_path=args.system_prompt_path)

    if args.do_train:
        remove_columns = get_remove_columns(os.path.join(args.data_dir, args.train_on))
        train_dataset = load_dataset("json", data_files=os.path.join(args.data_dir, args.train_on), split='train')
        train_dataset = train_dataset.shuffle(seed=args.seed)
        eval_dataset = load_dataset("json", data_files=os.path.join(args.data_dir, args.eval_on), split='train') if args.do_eval else None
        train_dataset = train_dataset.map(tokenize_row_train, remove_columns=remove_columns, num_proc=args.map_num_proc)
        #print(tokenizer.decode(train_dataset[0]['input_ids']))
        #print(train_dataset[0]['input_ids'][:800])
        #print(train_dataset[0]['attention_mask'][:800])
        #print(train_dataset[0]['labels'][:800])
        eval_dataset = eval_dataset.map(tokenize_row_train, remove_columns=remove_columns, num_proc=args.map_num_proc) if args.do_eval else None

        model = AutoModelForCausalLM.from_pretrained(args.load_model_path,
                                                    cache_dir=args.cache_dir,
                                                    trust_remote_code=True,
                                                    use_flash_attention_2=args.use_flash_attention_2,
                                                    torch_dtype=torch.bfloat16)
        
        if args.lora:
            if args.load_ckpt:
                model = PeftModel.from_pretrained(model, args.load_ckpt, is_trainable=True)
            else:
                peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1)
                model.enable_input_require_grads()
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            evaluation_strategy="epoch" if args.do_eval else 'no',
            logging_dir="logs",
            logging_strategy="epoch",
            save_strategy="no",
            fp16=args.fp16,
            deepspeed=args.deepspeed,
            local_rank=args.local_rank,
            gradient_checkpointing=args.gradient_checkpointing,
            bf16=args.bf16,
            # neftune_noise_alpha=args.adv_norm,
        )
        
        #train_dataset = TrainDataset(os.path.join(args.data_dir, args.train_on), args.max_seq_length, tokenizer, "train")
        #eval_dataset = TrainDataset(os.path.join(args.data_dir, args.eval_on), args.max_seq_length, tokenizer, "eval") if args.do_eval else None

        def compute_metrics(outputs):
            logits, labels = outputs
            predictions = logits.argmax(-1)
            labels = labels[:, 1:].reshape(-1)
            predictions = predictions[:, :-1].reshape(-1)
            return evaluate.load("accuracy", cache_dir='./metrics').compute(predictions=predictions, references=labels)

        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset if args.do_train else None,
            eval_dataset=eval_dataset if args.do_eval else None,
            compute_metrics=compute_metrics if args.do_eval else None,
        )

        train_result = trainer.train()
        #model = model.merge_and_unload()
        #model.save_pretrained(args.output_dir)
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_ppl"] = math.exp(metrics["train_loss"])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if args.do_test:
        accelerator = Accelerator()
        device = accelerator.device

        if args.lora:
            model = AutoModelForCausalLM.from_pretrained(args.load_model_path,
                                                        cache_dir=args.cache_dir,
                                                        use_flash_attention_2=args.use_flash_attention_2,
                                                        torch_dtype=torch.bfloat16)
            model = PeftModel.from_pretrained(model, args.load_ckpt)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.load_ckpt,
                                                        cache_dir=args.cache_dir,
                                                        use_flash_attention_2=args.use_flash_attention_2)

        remove_columns = get_remove_columns(os.path.join(args.data_dir, args.test_on))
        test_dataset = load_dataset("json", data_files=os.path.join(args.data_dir, args.test_on), split='train')   
        test_dataset = test_dataset.map(tokenize_row_test, remove_columns=remove_columns)
        #test_dataset = TrainDataset(os.path.join(args.data_dir, args.test_on), args.max_seq_length, tokenizer, "test")
        input_ids = torch.tensor(test_dataset['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(test_dataset['attention_mask'], dtype=torch.long)
        labels = torch.tensor(test_dataset['labels'], dtype=torch.long)
        test_dataset = TensorDataset(input_ids, attention_mask, labels)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.eval_batch_size)
        model, test_dataloader = accelerator.prepare(model, test_dataloader)
        model.eval()
        all_labels, all_predicts = [], []
        for batch in tqdm(test_dataloader, desc="Evaluation"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            with torch.no_grad():
                outputs = model.generate(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         max_new_tokens=args.max_new_tokens,
                                         eos_token_id=tokenizer.eos_token_id,
                                         pad_token_id=tokenizer.eos_token_id)
                padding_tensor = torch.LongTensor([[tokenizer.pad_token_id] * (args.max_seq_length + args.max_new_tokens - outputs.shape[1]) for _ in range(outputs.shape[0])]).to(outputs.device)
                predicts = torch.cat([outputs, padding_tensor], dim=1)
                #predicts = outputs
                #print(outputs.shape)

            labels, predicts = accelerator.gather_for_metrics((labels, predicts))
            all_labels.extend(labels.tolist())
            all_predicts.extend(predicts.tolist())
        
        if accelerator.is_local_main_process:
            ret = []
            for gt, rs in zip(all_labels, all_predicts):
                gt = [i for i in gt if i != -100]
                ret += [{"gt": tokenizer.decode(gt, skip_special_tokens=True).strip(), "rs": tokenizer.decode(rs[args.max_seq_length: ], skip_special_tokens=True).strip()}]
            
            compute(os.path.join(args.data_dir, args.test_on), args.results, ret)
def compute(test_path, result_path, ret):
    pass

if __name__ == "__main__":
    main()

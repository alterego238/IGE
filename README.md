# Game Development as Human-LLM Interaction

## environment
```
conda env create -f requirements.txt
```
The specific environment configuration can be found in the `requirements.txt` file. 
Our experiments were conducted on a machine with 8×A800 GPUs.

## data generation

### config openai_key
modify `utils.py` to config openai key

### pipeline
```bash
bash pipeline.sh
```

## train
```bash
bash sft.sh
bash sft_complete.sh
```

## eval
### eval ours
```bash
model=IGE_complete
CUDA_VISIBLE_DEVICES=0 python do_eval_interaction.py --model $model
python evaluate_interaction.py --model $model
python evaluate_wo_script.py --model $model
```

### eval all
config the `gpus` in `evaluate.sh` and then:
```bash
bash evaluate.sh
```
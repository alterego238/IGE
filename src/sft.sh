hostfile=""
HF_DATASETS_CACHE="../tmp" WANDB_MODE=disabled HF_ENDPOINT=https://hf-mirror.com deepspeed --include localhost:6,7 --master_port 21269 run_sft.py \
  --data_dir ../data \
  --do_train \
  --system_prompt_path ../prompt/system_prompt.md \
  --train_on train.jsonl \
  --load_model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --train_batch_size 2 \
  --eval_batch_size 1 \
  --num_train_epochs 3 \
  --max_seq_length 2000 \
  --bf16 \
  --learning_rate 3e-4 \
  --cache_dir ../../../cache \
  --output_dir ../model/IGE_stage1_test/ \
  --lora \
  --use_flash_attention_2 \
  --deepspeed ../deepspeed_configs/zero1_config.json \
  --map_num_proc 8 \
  --find_max_length #\
  #--load_ckpt ../model/IGE_stage1/ #\
  #--gradient_checkpointing

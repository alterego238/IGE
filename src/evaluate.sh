# api
models=("gpt-3.5-turbo" "gpt-4o-mini" "gpt-4o")
#models=()
for model in "${models[@]}"; do
    (
        echo "model: $model"
        python do_eval_interaction.py --model $model --api
        python evaluate_interaction.py --model $model
        python evaluate.py --model $model
    ) &
done

# local_no_ft
local_models=("llama3.1")
#local_models=()
gpus=("3")
length=${#local_models[@]}
for ((i=0; i<$length; i++)); do
    (
        model=${local_models[$i]}
        gpu=${gpus[$i]}
        
        echo "model: $model"
        CUDA_VISIBLE_DEVICES=6,7 python do_eval_interaction.py --model $model --no_ft --num_threads 4
        python evaluate_interaction.py --model $model
        python evaluate.py --model $model
    ) &
done

# local_no_stage0
local_models=("llama3.1_wo_instruct")
#local_models=()
gpus=("3")
length=${#local_models[@]}
for ((i=0; i<$length; i++)); do
    (
        model=${local_models[$i]}
        gpu=${gpus[$i]}
        
        echo "model: $model"
        CUDA_VISIBLE_DEVICES=6,7 python do_eval_interaction.py --model $model --wo_instruct --num_threads 4
        python evaluate_interaction.py --model $model
        python evaluate.py --model $model
    ) &
done

# local_wo_script
local_models=("wo_script_complete")
#local_models=()
gpus=("0")
length=${#local_models[@]}
for ((i=0; i<$length; i++)); do
    (
        model=${local_models[$i]}
        gpu=${gpus[$i]}
        
        echo "model: $model"
        CUDA_VISIBLE_DEVICES=$gpu python do_eval_interaction.py --model $model --wo_script
        python evaluate_interaction.py --model $model
        python evaluate_wo_script.py --model $model
    ) &
done

# local_ft
local_models=("IGE_wo_stage0")
#local_models=()
gpus=("1")
length=${#local_models[@]}
for ((i=0; i<$length; i++)); do
    (
        model=${local_models[$i]}
        gpu=${gpus[$i]}
        
        echo "model: $model"
        CUDA_VISIBLE_DEVICES=$gpu python do_eval_interaction.py --model $model --wo_stage0
        python evaluate_interaction.py --model $model
        python evaluate_wo_script.py --model $model
    ) &
done

# local_ft
local_models=("IGE_stage1" "IGE_stage2" "IGE_complete" "wo_synthesis_complete" "mixed")
#local_models=()
gpus=("1" "2" "3" "4" "5")
length=${#local_models[@]}
for ((i=0; i<$length; i++)); do
    (
        model=${local_models[$i]}
        gpu=${gpus[$i]}
        
        echo "model: $model"
        CUDA_VISIBLE_DEVICES=$gpu python do_eval_interaction.py --model $model
        python evaluate_interaction.py --model $model
        python evaluate_wo_script.py --model $model
    ) &
done


wait
echo "comeplete"

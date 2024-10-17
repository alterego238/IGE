python generate_script.py --split train
python get_seed_data.py --stage add_pool

for i in {1..5}
do
    python get_seed_data.py --stage seed_data
    python pipeline.py --stage code_script
done

python pipeline.py --stage interaction

python generate_script.py --split train_complete
python pipeline.py --stage interaction_complete
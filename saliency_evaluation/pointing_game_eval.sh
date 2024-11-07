export CUDA_VISIBLE_DEVICES=4
python -m saliency_evaluation.pointing_game_eval \
    --dataset_name="stanfordnlp/imdb" \
    --model_dir="models/bert_base_imdb_512" \
    --output_dir='baseline_results/all_methods_pointing_game_100_256' \
    --load_pointing_game_examples_path='baseline_results/all_methods_pointing_game_100_256/pointing_game_examples_256_100.json' \
    --save_pointing_game_examples_path='baseline_results/all_methods_pointing_game_100_256/pointing_game_examples_256_100.json' \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=256 \
    --num_examples=100 \

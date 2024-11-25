export CUDA_VISIBLE_DEVICES=1


python -m saliency_evaluation.bcos_pointing_game_eval \
    --dataset_name="stanfordnlp/imdb" \
    --model_dir="models/bcos_bert_base_imdb_512" \
    --output_dir='bcos_results/no_explanation_mode_pointing_game_imdb_only_token_pad_baselines' \
    --load_pointing_game_examples_path='pointing_game_data/all_orders_first.json' \
    --save_pointing_game_examples_path='pointing_game_data/all_orders_first.json' \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=256 \
    --num_examples=100 \


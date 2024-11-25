export CUDA_VISIBLE_DEVICES=6

embedding_types=("token" "position" "token_type")
embedding_types=("token" "position")

python -m saliency_evaluation.pointing_game_eval \
    --dataset_name="stanfordnlp/imdb" \
    --model_dir="models/bcos_bert_base_imdb_512" \
    --output_dir='bcos_results/pointing_game_imdb' \
    --load_pointing_game_examples_path='pointing_game_data/bcos_imdb_pointing_games.json' \
    --save_pointing_game_examples_path='pointing_game_data/bcos_imdb_pointing_games.json' \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=256 \
    --num_examples=100 \
    --baseline="pad" \
    --bcos \
    --b=2.0 \
    --relative \
#    --embedding_attributions "${embedding_types[@]}" \

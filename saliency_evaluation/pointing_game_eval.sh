export CUDA_VISIBLE_DEVICES=3

python -m saliency_evaluation.pointing_game_eval \
    --dataset_name="fancyzhx/ag_news" \
    --model_dir="models/bert_base_agnews_512_train" \
    --output_dir='baseline_results/pointing_game_agnews_train_100' \
    --load_pointing_game_examples_path='pointing_game_data/baseline_agnews_pointing_games_train_100.json' \
    --save_pointing_game_examples_path='pointing_game_data/baseline_agnews_pointing_games_train_100.json' \
    --num_labels=4 \
    --batch_size=1 \
    --max_length=50 \
    --num_examples=100 \
    --baseline="pad" \
    --methods "Attention, Saliency, DeepLift, InputXGradient, Occlusion, KernelShap, ShapleyValue, Lime" \


python -m saliency_evaluation.pointing_game_eval \
    --dataset_name="stanfordnlp/imdb" \
    --model_dir="models/bert_base_imdb_512_train" \
    --output_dir='baseline_results/pointing_game_imdb_train_100' \
    --load_pointing_game_examples_path='pointing_game_data/baseline_imdb_pointing_games_train_100.json' \
    --save_pointing_game_examples_path='pointing_game_data/baseline_imdb_pointing_games_train_100.json' \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=256 \
    --num_examples=100 \
    --baseline="pad" \
    --methods "Lime, IntegratedGradients" \


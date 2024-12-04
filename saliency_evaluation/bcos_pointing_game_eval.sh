export CUDA_VISIBLE_DEVICES=6


python -m saliency_evaluation.pointing_game_eval \
    --dataset_name="stanfordnlp/imdb" \
    --model_dir="models/bcos_bert_base_imdb_512_bce_no_embedding_norm_gelu_new_implementation_train" \
    --output_dir='bcos_results/pointing_game_imdb_bce_no_embedding_norm_gelu_new_implementation_train_100' \
    --load_pointing_game_examples_path='pointing_game_data/bcos_imdb_pointing_games_bce_no_embedding_norm_gelu_new_implementation_train_100.json' \
    --save_pointing_game_examples_path='pointing_game_data/bcos_imdb_pointing_games_bce_no_embedding_norm_gelu_new_implementation_train_100.jso' \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=256 \
    --num_examples=100 \
    --baseline="pad" \
    --methods "Bcos, Attention, Saliency, DeepLift, InputXGradient, Occlusion, KernelShap, ShapleyValue, Lime" \

python -m saliency_evaluation.pointing_game_eval \
    --dataset_name="fancyzhx/ag_news" \
    --model_dir="models/bcos_bert_base_agnews_512_bce_no_embedding_norm_gelu_new_implementation_train" \
    --output_dir='bcos_results/pointing_game_agnews_bce_no_embedding_norm_gelu_new_implementation_train_100' \
    --load_pointing_game_examples_path='pointing_game_data/bcos_agnews_pointing_games_bce_no_embedding_norm_gelu_new_implementation_train_100.jso' \
    --save_pointing_game_examples_path='pointing_game_data/bcos_agnews_pointing_games_bce_no_embedding_norm_gelu_new_implementation_train_100.jso' \
    --num_labels=4 \
    --batch_size=1 \
    --max_length=50 \
    --num_examples=100 \
    --baseline="pad" \
    --methods "Bcos, Attention, Saliency, DeepLift, InputXGradient, Occlusion, KernelShap, ShapleyValue, Lime" \
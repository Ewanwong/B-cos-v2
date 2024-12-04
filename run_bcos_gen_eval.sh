export CUDA_VISIBLE_DEVICES=4

python -m saliency_generation.gen_explanations \
    --dataset_name="stanfordnlp/imdb" \
    --model_dir="models/bcos_bert_base_imdb_512_bce_no_embedding_norm_gelu_new_implementation_train" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods "Bcos, Attention, Saliency, DeepLift, InputXGradient, Occlusion, KernelShap, ShapleyValue, Lime, IntegratedGradients" \
    --output_dir="bcos_results/perturbation_imdb_absolute_bce_no_embedding_norm_gelu_new_implementation_train_100" \
    --only_predicted_classes \
    --baseline="pad" \



python -m saliency_evaluation.perturbation_eval \
    --explanation_dir="bcos_results/perturbation_imdb_absolute_bce_no_embedding_norm_gelu_new_implementation_train_100" \
    --model_dir="models/bcos_bert_base_imdb_512_bce_no_embedding_norm_gelu_new_implementation_train" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=-1 \



python -m saliency_generation.gen_explanations \
    --dataset_name="fancyzhx/ag_news" \
    --model_dir="models/bcos_bert_base_agnews_512_bce_no_embedding_norm_gelu_new_implementation_train" \
    --num_labels=4 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods "Bcos, Attention, Saliency, DeepLift, InputXGradient, Occlusion, KernelShap, ShapleyValue, Lime, IntegratedGradients" \
    --output_dir="bcos_results/perturbation_agnews_absolute_bce_no_embedding_norm_gelu_new_implementation_train_100" \
    --only_predicted_classes \
    --baseline="pad" \



python -m saliency_evaluation.perturbation_eval \
    --explanation_dir="bcos_results/perturbation_agnews_absolute_bce_no_embedding_norm_gelu_new_implementation_train_100" \
    --model_dir="models/bcos_bert_base_agnews_512_bce_no_embedding_norm_gelu_new_implementation_train" \
    --num_labels=4 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=-1 \


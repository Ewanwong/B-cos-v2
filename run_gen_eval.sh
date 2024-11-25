export CUDA_VISIBLE_DEVICES=1
python -m saliency_generation.gen_explanations \
    --dataset_name="stanfordnlp/imdb" \
    --model_dir="models/bert_base_imdb_512" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --output_dir="baseline_results/perturbation_imdb_zero_baseline" \
    --only_predicted_classes \
    --baseline="zero" \

python -m saliency_evaluation.perturbation_eval \
    --explanation_dir="baseline_results/perturbation_imdb_zero_baseline" \
    --model_dir="models/bert_base_imdb_512" \
    --num_labels=1 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=-1 \

python -m saliency_generation.gen_explanations \
    --dataset_name="fancyzhx/ag_news" \
    --model_dir="models/bert_base_agnews_512" \
    --num_labels=4 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --output_dir="baseline_results/perturbation_agnews_zero_baseline" \
    --only_predicted_classes \
    --baseline="zero" \

python -m saliency_evaluation.perturbation_eval \
    --explanation_dir="baseline_results/perturbation_agnews_zero_baseline" \
    --model_dir="models/bert_base_agnews_512" \
    --num_labels=4 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=-1 \
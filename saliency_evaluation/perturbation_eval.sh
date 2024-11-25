export CUDA_VISIBLE_DEVICES=2
python -m saliency_evaluation.perturbation_eval \
    --explanation_dir="baseline_results/perturbation_imdb" \
    --model_dir="models/bert_base_imdb_512" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=-1 \


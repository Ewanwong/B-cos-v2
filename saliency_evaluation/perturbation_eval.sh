export CUDA_VISIBLE_DEVICES=0
python -m saliency_evaluation.perturbation_eval \
    --explanation_dir="baseline_saliency_results/all_methods_1000_examples_512" \
    --model_dir="models/bert_base_imdb_512" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=-1 \


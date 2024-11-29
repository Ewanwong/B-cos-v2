export CUDA_VISIBLE_DEVICES=0
python -m saliency_evaluation.human_agreement_eval \
    --model_dir="models/bert_base_imdb_512" \
    --batch_size=1 \
    --max_length=512 \
    --baseline 'pad' \
    --output_dir="baseline_results/human_agreement" \
    --seed=42 \
    --only_predicted_classes \
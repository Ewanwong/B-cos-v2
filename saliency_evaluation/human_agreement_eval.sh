export CUDA_VISIBLE_DEVICES=3
python -m saliency_evaluation.human_agreement_eval \
    --model_dir="models/bert_base_imdb_512_train" \
    --batch_size=1 \
    --max_length=512 \
    --baseline 'pad' \
    --output_dir="baseline_results/human_agreement_train" \
    --seed=42 \
    --only_predicted_classes \
    --methods "Attention, Saliency, DeepLift, InputXGradient, Occlusion, KernelShap, ShapleyValue, Lime, IntegratedGradients" \
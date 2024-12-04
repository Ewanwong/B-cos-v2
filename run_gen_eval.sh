export CUDA_VISIBLE_DEVICES=3
#python -m saliency_generation.gen_explanations \
#    --dataset_name="stanfordnlp/imdb" \
#    --model_dir="models/bert_base_imdb_512_train" \
#    --num_labels=2 \
#    --batch_size=1 \
#    --max_length=512 \
#    --num_examples=100 \
#    --output_dir="baseline_results/perturbation_imdb_train" \
#    --only_predicted_classes \
#    --baseline="pad" \
#    --methods "Attention, Saliency, DeepLift, InputXGradient, Occlusion, KernelShap, ShapleyValue, Lime" \

python -m saliency_evaluation.perturbation_eval \
    --explanation_dir="baseline_results/perturbation_imdb_train" \
    --model_dir="models/bert_base_imdb_512_train" \
    --num_labels=2 \
    --batch_size=4 \
    --max_length=512 \
    --num_examples=-1 \

python -m saliency_generation.gen_explanations \
    --dataset_name="fancyzhx/ag_news" \
    --model_dir="models/bert_base_agnews_512_train" \
    --num_labels=4 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --output_dir="baseline_results/perturbation_agnews_train" \
    --only_predicted_classes \
    --baseline="pad" \
    --methods "Attention, Saliency, DeepLift, InputXGradient, Occlusion, KernelShap, ShapleyValue, Lime" \

python -m saliency_evaluation.perturbation_eval \
    --explanation_dir="baseline_results/perturbation_agnews_train" \
    --model_dir="models/bert_base_agnews_512_train" \
    --num_labels=4 \
    --batch_size=4 \
    --max_length=512 \
    --num_examples=-1 \
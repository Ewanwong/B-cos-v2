export CUDA_VISIBLE_DEVICES=2

python -m saliency_evaluation.perturbation_eval \
    --explanation_dir="bcos_results/perturbation_imdb_absolute" \
    --model_dir="models/bcos_bert_base_imdb_512" \
    --num_labels=2 \
    --batch_size=4 \
    --max_length=512 \
    --num_examples=-1 \
    #--bcos \
    #--b=2.0 \


python -m saliency_evaluation.perturbation_eval \
    --explanation_dir="bcos_results/perturbation_agnews_absolute" \
    --model_dir="models/bcos_bert_base_agnews_512" \
    --num_labels=4 \
    --batch_size=4 \
    --max_length=512 \
    --num_examples=-1 \
    #--bcos \
    #--b=2.0 \



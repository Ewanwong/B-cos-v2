export CUDA_VISIBLE_DEVICES=0
python -m saliency_generation.gen_explanations \
    --dataset_name="stanfordnlp/imdb" \
    --model_dir="models/bert_base_imdb_512" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --output_dir="baseline_explanations/try" \
    

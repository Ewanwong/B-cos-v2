export CUDA_VISIBLE_DEVICES=2
python -m bert_finetuning \
    --model_name_or_path "bert-base-uncased" \
    --dataset_name="fancyzhx/ag_news" \
    --num_labels=4 \
    --output_dir "/local/yifwang/bcos_bert_base_agnews_512_bce" \
    --batch_size=16 \
    --max_seq_length=512 \
    --learning_rate=3e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=10 \
    --early_stopping_patience=-1 \
    --seed=42 \
    --b 2.0 \
    --bcos \
    --bce \


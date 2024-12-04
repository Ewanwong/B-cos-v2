export CUDA_VISIBLE_DEVICES=3






python -m bert_finetuning \
    --model_name_or_path "bert-base-uncased" \
    --dataset_name="fancyzhx/ag_news" \
    --num_labels=4 \
    --output_dir "/local/yifwang/bert_base_agnews_512_train" \
    --batch_size=16 \
    --max_seq_length=512 \
    --learning_rate=3e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --seed=42 \





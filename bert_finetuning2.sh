export CUDA_VISIBLE_DEVICES=4

python -m bert_finetuning \
    --model_name_or_path "bert-base-uncased" \
    --dataset_name="stanfordnlp/imdb" \
    --num_labels=2 \
    --output_dir "/local/yifwang/bcos_bert_base_imdb_512_bce_no_embedding_norm_gelu_new_implementation_train" \
    --batch_size=16 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --seed=42 \
    --b 2.0 \
    --bcos \
    --bce \
    --no_embedding_norm \

python -m bert_finetuning \
    --model_name_or_path "bert-base-uncased" \
    --dataset_name="fancyzhx/ag_news" \
    --num_labels=4 \
    --output_dir "/local/yifwang/bcos_bert_base_agnews_512_bce_no_embedding_norm_gelu_new_implementation_train" \
    --batch_size=16 \
    --max_seq_length=512 \
    --learning_rate=3e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --seed=42 \
    --b 2.0 \
    --bcos \
    --bce \
    --no_embedding_norm \

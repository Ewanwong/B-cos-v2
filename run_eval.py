import argparse
import json
import logging
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from transformers import (BertTokenizer, AutoConfig, AdamW,
                          get_linear_schedule_with_warmup)
from bcos_lm.models.modeling_bert import BertForSequenceClassification
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
import random
import os
from tqdm import tqdm

def main():
    # Argument parser for hyperparameters
    parser = argparse.ArgumentParser(description="Fine-tune BERT for sequence classification")

    # Hyperparameters
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased',
                        help='Pre-trained model name or path')
    parser.add_argument('--dataset_name', type=str, default='stanfordnlp/imdb',
                        help='Dataset name (default: ag_news)')
    parser.add_argument('--num_labels', type=int, default=2,
                        help='Number of labels in the dataset')
    parser.add_argument('--output_dir', type=str, default='models/bcos_bert_base_imdb_512',
                        help='Directory to save the model')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Maximum input sequence length after tokenization')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='Learning rate for the optimizer')
    parser.add_argument('--warmup_steps_or_ratio', type=int, default=0.1,
                        help='Number or ratio of warmup steps for the learning rate scheduler')
    parser.add_argument('--num_train_epochs', type=int, default=10,
                        help='Total number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for initialization')
    parser.add_argument('--early_stopping_patience', type=int, default=-1,
                        help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--log_file', type=str, default='training.log',
                        help='Path to the log file')
    parser.add_argument('--eval_steps', type=int, default=2000,
                        help='Evaluate the model every X training steps')
    parser.add_argument('--save_steps', type=int, default=2000,
                        help='Save the model every X training steps')
    parser.add_argument('--split_ratio', type=float, default=0.5,
                    help='Ratio to split the test set into validation and test sets')
    parser.add_argument('--b', type=float, default=2.0,)
    parser.add_argument('--bcos', action='store_true', help='Use BCOS')
    args = parser.parse_args()

    # create output directory if it doesn't exist
    
    # Set up the device for GPU usage if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Set seeds for reproducibility
    seed_val = args.seed

    def set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_random_seed(seed_val)


    # Load the dataset
    logging.info(f"Loading {args.dataset_name} dataset...")
    dataset = load_dataset(args.dataset_name)
    
    # Initialize the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
    config.num_labels = args.num_labels
    config.bcos = True
    config.b = args.b


    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples['text'],
                         padding='max_length',
                         truncation=True,
                         max_length=args.max_seq_length)


    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Set the format of the datasets to PyTorch tensors
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    # change the column "label" to "labels" to match the model's forward function
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')


    # Prepare data loaders
    train_dataset = tokenized_datasets['train']
    test_dataset = tokenized_datasets['test']

   # Split the test dataset into validation and test sets
    test_dataset_size = len(test_dataset)
    indices = list(range(test_dataset_size))
    split = int(np.floor(args.split_ratio * test_dataset_size))
    np.random.shuffle(indices)

    val_indices, test_indices = indices[:split], indices[split:]

    val_dataset = Subset(test_dataset, val_indices)
    test_dataset = Subset(test_dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size)
    validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.batch_size)

    # Initialize the optimizer and learning rate scheduler
   

    # Accuracy evaluation function
    def evaluate(model, dataloader):
        model.eval()
        predictions, true_labels = [], []

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits.detach().cpu().numpy()
            label_ids = batch['labels'].to('cpu').numpy()

            predictions.extend(np.argmax(logits, axis=1))
            true_labels.extend(label_ids)

        accuracy = accuracy_score(true_labels, predictions)
        return accuracy


    # Load the best model
    model = BertForSequenceClassification.load_from_pretrained(args.output_dir, config=config)
    tokenizer = BertTokenizer.from_pretrained(args.output_dir)
    model.to(device)

    # Test evaluation

    test_accuracy = evaluate(model, test_dataloader)
    print(test_accuracy)

    val_accuracy = evaluate(model, validation_dataloader)
    print(val_accuracy)

if __name__ == '__main__':
    main()
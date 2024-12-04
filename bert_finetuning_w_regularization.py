import argparse
import json
import math
import logging
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from transformers import (BertTokenizer, AutoConfig, AdamW,
                          get_linear_schedule_with_warmup)
from bcos_lm.models.new_modeling_bert import BertForSequenceClassification
from saliency_utils.Explainer import BertEmbeddingModelWrapper
from bcos_lm.losses import ConsecutiveLoss
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
    parser.add_argument('--dataset_name', type=str, default='fancyzhx/ag_news',
                        help='Dataset name (default: ag_news)')
    parser.add_argument('--num_labels', type=int, default=4,
                        help='Number of labels in the dataset')
    parser.add_argument('--output_dir', type=str, default='/local/yifwang/bcos_bert_base_agnews_512',
                        help='Directory to save the model')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Maximum input sequence length after tokenization')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='Learning rate for the optimizer')
    parser.add_argument('--warmup_steps_or_ratio', type=float, default=0.1,
                        help='Number or ratio of warmup steps for the learning rate scheduler')
    parser.add_argument('--num_train_epochs', type=int, default=10,
                        help='Total number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for initialization')
    parser.add_argument('--early_stopping_patience', type=int, default=-1,
                        help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--log_file', type=str, default='training.log',
                        help='Path to the log file')
    parser.add_argument('--eval_steps', type=int, default=1000,
                        help='Evaluate the model every X training steps')
    parser.add_argument('--save_steps', type=int, default=1000,
                        help='Save the model every X training steps')
    parser.add_argument('--split_ratio', type=float, default=0.5,
                    help='Ratio to split the test set into validation and test sets')
    parser.add_argument('--b', type=float, default=2.0,)
    parser.add_argument('--bcos', action='store_true', help='Use BCOS')
    parser.add_argument('--bce', action='store_true', help='Use bce loss instead of cross entropy loss')
    parser.add_argument('--relative_logits', action='store_true', help='Use relative logit')
    parser.add_argument('--bcos_attention', action='store_true', help='Use BCOS attention')
    parser.add_argument('--no_embedding_norm', action='store_true', help='Do not normalize the embeddings')
    parser.add_argument('--alpha', type=float, default=0.1, help='Regularization coefficient')
    parser.add_argument('--reg_loss', type=str, default='auto_corr', help='Regularization loss type')
    args = parser.parse_args()

    # create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    log_file = os.path.join(args.output_dir, args.log_file)

    # Set up logging
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # Log the hyperparameters
    logging.info("Hyperparameters:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

    # Set up the device for GPU usage if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

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
    config.bcos = args.bcos
    config.b = args.b
    config.bce = args.bce
    config.relative_logits = args.relative_logits
    config.bcos_attention = args.bcos_attention
    config.no_embedding_norm = args.no_embedding_norm
    model = BertForSequenceClassification.load_from_pretrained(args.model_name_or_path, config=config)
    model.to(device)
    embedding_model = BertEmbeddingModelWrapper(model)

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples['text'],
                         padding='max_length',
                         truncation=True,
                         max_length=args.max_seq_length)

    # Apply tokenization to the datasets
    logging.info("Tokenizing datasets...")
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
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    total_steps = len(train_dataloader) * args.num_train_epochs
    if args.warmup_steps_or_ratio > 1.0:
        warmup_steps = args.warmup_steps_or_ratio
    else:
        warmup_steps = int(total_steps * args.warmup_steps_or_ratio)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    # initialize the loss function
    reg_loss_fn = ConsecutiveLoss(args.reg_loss)
    task_loss_fn = torch.nn.BCELoss() if args.bce else torch.nn.CrossEntropyLoss()
    sigmoid = torch.nn.Sigmoid()

    # Accuracy evaluation function
    def evaluate(model, dataloader):
        model.eval()
        embedding_model = BertEmbeddingModelWrapper(model)
        predictions, true_labels = [], []
        regularization_losses = []

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            embeddings = model.bert.embeddings(input_ids=batch['input_ids'])
            embeddings.requires_grad_()
            outputs = embedding_model(embeddings, attention_mask=batch['attention_mask'])

            logits = outputs.detach().cpu().numpy()
            label_ids = batch['labels'].to('cpu').numpy()

            predictions.extend(np.argmax(logits, axis=1))
            true_labels.extend(label_ids)

            with embedding_model.model.explanation_mode():
                target_outputs = torch.gather(outputs, 1, batch['labels'].view(-1, 1))
                grads = torch.autograd.grad(torch.unbind(target_outputs), embeddings, create_graph=True, retain_graph=True)[0]
                attributions = (grads * embeddings).sum(dim=-1)
                reg_loss = reg_loss_fn(attributions)
                regularization_losses.append(reg_loss.detach().item())
        val_reg_loss = np.mean(regularization_losses)
        accuracy = accuracy_score(true_labels, predictions)
        model.train()
        return accuracy, val_reg_loss

    # Early stopping parameters
    early_stopping_patience = args.early_stopping_patience if args.early_stopping_patience != -1 else np.inf
    best_accuracy = 0.0
    evaluations_no_improve = 0
    global_step = 0
    
    # Training loop
    for epoch_i in range(args.num_train_epochs):
        logging.info(f"\n======== Epoch {epoch_i + 1} / {args.num_train_epochs} ========")
        logging.info("Training...")

        total_loss = 0
        total_task_loss = 0
        total_reg_loss = 0
        model.train()

        for step, batch in tqdm(enumerate(train_dataloader)):

            global_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels']
            optimizer.zero_grad()

            embeddings = model.bert.embeddings(input_ids=batch['input_ids'])
            embeddings.requires_grad_()
            outputs = embedding_model(embeddings, attention_mask=batch['attention_mask'])

            # compute task loss
            if not args.bce:
                task_loss = task_loss_fn(outputs.view(-1, config.num_labels), labels.view(-1))
            else:
                targets = torch.nn.functional.one_hot(labels, num_classes=config.num_labels).float()
                targets.requires_grad = False
                if not args.relative_logits:
                    task_loss = task_loss_fn(sigmoid(outputs), targets)
                else:
                    task_loss = task_loss_fn(sigmoid(outputs+math.log(1/(config.num_labels-1))), targets)
            total_task_loss += task_loss.item()

            # compute regularization loss
            with model.explanation_mode():
                target_outputs = torch.gather(outputs, 1, batch['labels'].view(-1, 1))
                grads = torch.autograd.grad(torch.unbind(target_outputs), embeddings, create_graph=True, retain_graph=True)[0]
                attributions = (grads * embeddings).sum(dim=-1)
                reg_loss = reg_loss_fn(attributions)

            total_reg_loss += reg_loss.item()
            loss = task_loss + args.alpha * reg_loss
            total_loss += loss.detach().item()
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Evaluate the model at specified steps
            if args.eval_steps and global_step % args.eval_steps == 0:
                logging.info(f"\nStep {global_step}: running evaluation...")
                val_accuracy, val_reg_loss = evaluate(model, validation_dataloader)
                logging.info(f"Validation Accuracy at step {global_step}: {val_accuracy:.4f}")
                logging.info(f"Validation Regularization Loss at step {global_step}: {val_reg_loss:.4f}")

                # Check for early stopping
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    evaluations_no_improve = 0

                    # Save the best model using Hugging Face's save_pretrained
                    model.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
                    logging.info(f"Best model saved to {args.output_dir}")
                else:
                    evaluations_no_improve += 1
                    logging.info(f"No improvement in validation accuracy for {evaluations_no_improve} evaluation(s).")
                    if evaluations_no_improve >= early_stopping_patience:
                        logging.info("Early stopping triggered.")
                        break

            # Save the model at specified steps
            if args.save_steps and global_step % args.save_steps == 0:
                checkpoint_dir = os.path.join(args.output_dir, f'checkpoint-{global_step}')
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                logging.info(f"Model checkpoint saved at step {global_step} to {checkpoint_dir}")

        avg_train_loss = total_loss / len(train_dataloader)
        logging.info(f"Average training loss for epoch {epoch_i + 1}: {avg_train_loss:.4f}")

        # Evaluate at the end of each epoch if eval_steps is not set
        if not args.eval_steps:
            logging.info("Running Validation at the end of the epoch...")
            val_accuracy, val_reg_loss = evaluate(model, validation_dataloader)
            logging.info(f"Validation Accuracy: {val_accuracy:.4f}")
            logging.info(f"Validation Regularization Loss: {val_reg_loss:.4f}")

            # Check for early stopping
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                evaluations_no_improve = 0

                # Save the best model using Hugging Face's save_pretrained
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                logging.info(f"Best model saved to {args.output_dir}")
            else:
                evaluations_no_improve += 1
                logging.info(f"No improvement in validation accuracy for {evaluations_no_improve} epoch(s).")
                if evaluations_no_improve >= early_stopping_patience:
                    logging.info("Early stopping triggered.")
                    break

        # Save the model at the end of each epoch if save_steps is not set
        if not args.save_steps:
            checkpoint_dir = os.path.join(args.output_dir, f'checkpoint-epoch-{epoch_i + 1}')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            logging.info(f"Model checkpoint saved at the end of epoch {epoch_i + 1} to {checkpoint_dir}")

        # Break the outer loop if early stopping is triggered during evaluation steps
        if evaluations_no_improve >= early_stopping_patience:
            break

    # Load the best model
    model = BertForSequenceClassification.load_from_pretrained(args.output_dir, config=config)
    tokenizer = BertTokenizer.from_pretrained(args.output_dir)
    model.to(device)

    # Test evaluation
    logging.info("\nRunning Test Evaluation...")
    test_accuracy, test_reg_loss = evaluate(model, test_dataloader)
    logging.info(f"Test Accuracy: {test_accuracy:.4f}")
    logging.info(f"Test Regularization Loss: {test_reg_loss:.4f}")

if __name__ == '__main__':
    main()
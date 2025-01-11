from bcos_lm.models.new_new_modeling_bert import BertForMaskedLM
from bcos_lm.models.new_new_modeling_roberta import RobertaForMaskedLM
from bcos_lm.models.new_new_modeling_distilbert import DistilBertForMaskedLM


from transformers import AutoTokenizer, AutoConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, TrainerCallback
import torch
import time
import argparse
import logging
import random
from datasets import load_dataset
# import glob
# import csv
# import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
# from tqdm.notebook import tqdm
# tqdm.pandas()




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
    parser.add_argument('--mlm_prob', type=float, default=0.15, help='mlm masking probability')
    parser.add_argument('--b', type=float, default=2.0,)
    parser.add_argument('--bcos', action='store_true', help='Use BCOS')
    parser.add_argument('--bce', action='store_true', help='Use bce loss instead of cross entropy loss')
    parser.add_argument('--different_b_per_layer', action='store_true', help='Use different b per layer')
    parser.add_argument('--b_list', type=str, default="1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0", help='List of b values for different layers')


    args = parser.parse_args()
    print("start experiment")
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


    num_train = 25000
    num_val = 12500
    num_test = 12500
    # Load the dataset


    logging.info(f"Loading {args.dataset_name} dataset...")
    dataset = load_dataset(args.dataset_name, "20231101.en")['train']
    train_dataset = dataset.select(range(num_train))
    val_dataset = dataset.select(range(num_train, num_train+num_val))
    test_dataset = dataset.select(range(num_train+num_val, num_train+num_val+num_test))
    # Create Masked Language Model
    print("dataset loaded")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)


    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples['text'],
                         padding='max_length',
                         truncation=True,
                         max_length=512)


    # Apply tokenization to the datasets
    logging.info("Tokenizing datasets...")
    tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True)


    # Set the format of the datasets to PyTorch tensors
    tokenized_train_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])


    tokenized_eval_datasets = val_dataset.map(tokenize_function, batched=True)
    tokenized_eval_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])








    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob
    )


    # Initialize the tokenizer and model
    if "distilbert" in args.model_name_or_path.lower():
        Model = DistilBertForMaskedLM
    elif "roberta" in args.model_name_or_path.lower():
        Model = RobertaForMaskedLM
    elif "bert" in args.model_name_or_path.lower():
        Model = BertForMaskedLM
    else:
        raise ValueError("Model not supported")
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.bcos = args.bcos
    config.b = args.b
    config.bce = args.bce
    config.different_b_per_layer = args.different_b_per_layer
    b_list = [float(b) for b in args.b_list.strip().split(",")] if args.different_b_per_layer else [args.b] * config.num_hidden_layers
    assert len(b_list) == config.num_hidden_layers if hasattr(config, "num_hidden_layers") else config.n_layers, "Length of b_list should be equal to the number of hidden layers"
    config.b_list = b_list
    model = Model.load_from_pretrained(args.model_name_or_path, config=config)
    model.to(device)
    print("model loaded")
    # Create lamda tokenizing function
    def map_tokenize(text):
        return tokenizer.encode(text, max_length=args.sequence_len, truncation=True)


    warmup_steps = int(args.warmup_steps_or_ratio * len(train_dataset) // args.batch_size * args.num_train_epochs)
    print("start training")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        load_best_model_at_end=False,    
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=args.eval_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        prediction_loss_only=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=warmup_steps,
    )


    class MyTrainer(Trainer):
        def _save(self, output_dir=None, state_dict=None):
           
            """
            Override the default _save method to disable safe serialization.
            """
            if output_dir is None:
                output_dir = self.args.output_dir


            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Saving model checkpoint to {output_dir}")


            # Save the model, explicitly disabling safe serialization:
            if not self.deepspeed:
                self.model.save_pretrained(
                    output_dir, safe_serialization=False, state_dict=state_dict
                )
            else:
                # Deepspeed has its own mechanism, but you can often do the same:
                self.model.save_pretrained(output_dir, safe_serialization=False)


            # Save tokenizer if present
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)


    class SkipFinalStepCallback(TrainerCallback):
        """
        A callback that prevents saving and evaluating at the final step (end of last epoch).
        """
        def on_evaluate(self, args, state, control, **kwargs):
            # If we're at the end of the last epoch, skip evaluation
            if state.epoch is not None and state.epoch >= args.num_train_epochs:
                control.should_evaluate = False


        def on_save(self, args, state, control, **kwargs):
            # If we're at the end of the last epoch, skip saving
            if state.epoch is not None and state.epoch >= args.num_train_epochs:
                control.should_save = False
   
    trainer = MyTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_train_datasets,
        eval_dataset=tokenized_eval_datasets,
        callbacks=[SkipFinalStepCallback()],
    )


    trainer.train()


    # 3. Identify the best checkpoint (Trainer tracked it during training)
    best_checkpoint = trainer.state.best_model_checkpoint
    print("Best checkpoint is:", best_checkpoint)
    #print("Best step:", trainer.state.best_step)
    logging.info(f"Best checkpoint is: {best_checkpoint}")
    #logging.info(f"Best step: {trainer.state.best_step}")


    # copy everything from the best checkpoint to the output directory
    os.system(f"cp -r {best_checkpoint}/* {args.output_dir}")


    # save config
    config.save_pretrained(args.output_dir)
    # save tokenizer
    tokenizer.save_pretrained(args.output_dir)



if __name__ == "__main__":
    main()






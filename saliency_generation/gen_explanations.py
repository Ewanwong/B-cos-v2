from saliency_utils.Explainer import AttentionExplainer, GradientNPropabationExplainer, OcclusionExplainer, ShapleyValueExplainer, LimeExplainer
from saliency_utils.utils import set_random_seed, split_dataset
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import numpy as np
import json
import os
import random
from tqdm import tqdm

EXPLANATION_METHODS = {
    "Attention": AttentionExplainer,
    "Saliency": GradientNPropabationExplainer,
    "DeepLift": GradientNPropabationExplainer,
    "GuidedBackprop": GradientNPropabationExplainer,
    "InputXGradient": GradientNPropabationExplainer,
    "IntegratedGradients": GradientNPropabationExplainer,
    "Occlusion": OcclusionExplainer,
    "ShapleyValue": ShapleyValueExplainer,
    "Lime": LimeExplainer,
}


def main(args):
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load pre-trained BERT model and tokenizer
    model = BertForSequenceClassification.from_pretrained(args.model_dir, output_attentions=True)
    model.eval()
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained(args.model_dir)

    # Load a dataset from HuggingFace datasets library
    dataset = load_dataset(args.dataset_name, split=args.split)
    
    # If num_examples is specified, take a subset of the dataset
    if args.split_ratio is not None:
        dataset = split_dataset(dataset, args.split_ratio)[:args.num_examples]

    # add column index to the dataset
    dataset['index'] = list(range(len(dataset['text'])))

    # Initialize the explainer
    all_methods = EXPLANATION_METHODS.keys()
    if args.methods:
        attribution_methods = args.methods   
    else:
        attribution_methods = all_methods  # Use all methods if none specified


    # Create output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for method in attribution_methods:
        print(f"\nRunning {method} explainer...")
        if method == "ShapleyValue":
            explainer = ShapleyValueExplainer(model, tokenizer, args.n_samples)
        # for GradientNPropabationExplainer, we need to specify the method
        elif EXPLANATION_METHODS[method] == GradientNPropabationExplainer:
            explainer = EXPLANATION_METHODS[method](model, tokenizer, method)
        else:
            explainer = EXPLANATION_METHODS[method](model, tokenizer) 

        # can only explain the label class to reduce the computation time
        #class_labels = [dataset['label']]
        #explanation_results = explainer.explain_dataset(dataset, num_classes=args.num_labels, class_labels=class_labels, batch_size=args.batch_size, max_length=args.max_length)
        
        explanation_results = explainer.explain_dataset(dataset, num_classes=args.num_labels, batch_size=args.batch_size, max_length=args.max_length)
        result = explanation_results

        # Save the results to a JSON file
        output_file = os.path.join(args.output_dir, f'{method}_explanations.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"\nAttribution results saved to {output_file}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT Attribution with Captum')

    parser.add_argument('--dataset_name', type=str, default='stanfordnlp/imdb', help='Name of the HuggingFace dataset to use') #fancyzhx/ag_news, stanfordnlp/imdb
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, test)')
    parser.add_argument('--split_ratio', type=float, default=0.5, help='Split ratio for test dataset')
    parser.add_argument('--model_dir', type=str, default='models/bert_base_imdb_512', help='Name of the pre-trained model')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels in the classification')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length for tokenization')
    parser.add_argument('--num_examples', type=int, default=1000, help='Number of examples to process (-1 for all)')
    parser.add_argument('--methods', nargs='+', default=None, help='List of attribution methods to use')
    parser.add_argument('--output_dir', type=str, default='baseline_saliency_results/all_methods_1000_examples_512', help='Directory to save the output files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--n_samples', type=int, default=25, help='Number of samples for Shapley Value Sampling')

    args = parser.parse_args()
    main(args)

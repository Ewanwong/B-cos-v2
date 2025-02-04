from saliency_utils.Explainer import BcosExplainer, AttentionExplainer, GradientNPropabationExplainer, OcclusionExplainer, ShapleyValueExplainer, LimeExplainer
from saliency_utils.utils import set_random_seed, split_dataset
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoConfig
from bcos_lm.models.new_new_modeling_bert import BertForSequenceClassification
from bcos_lm.models.new_new_modeling_roberta import RobertaForSequenceClassification
from bcos_lm.models.new_new_modeling_distilbert import DistilBertForSequenceClassification
from datasets import load_dataset
import numpy as np
import json
import os
import random
from tqdm import tqdm

EXPLANATION_METHODS = {
    "Bcos": BcosExplainer,
    "Attention": AttentionExplainer,
    "Saliency": GradientNPropabationExplainer,
    "DeepLift": GradientNPropabationExplainer,
    #"GuidedBackprop": GradientNPropabationExplainer,
    "InputXGradient": GradientNPropabationExplainer,
    "IntegratedGradients": GradientNPropabationExplainer,
    "SIG": GradientNPropabationExplainer,
    "Occlusion": OcclusionExplainer,
    "ShapleyValue": ShapleyValueExplainer,
    "KernelShap": ShapleyValueExplainer,
    "Lime": LimeExplainer,
}


def main(args):

    # convert strings to numbers
    args.num_labels = int(args.num_labels) if args.num_labels else None
    args.batch_size = int(args.batch_size) if args.batch_size else None
    args.max_length = int(args.max_length) if args.max_length else None
    args.num_examples = int(args.num_examples) if args.num_examples else None
    args.seed = int(args.seed) if args.seed else None
    args.shap_n_samples = int(args.shap_n_samples) if args.shap_n_samples else None


    # Set random seed for reproducibility
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load pre-trained BERT model and tokenizer
    if "distilbert" in args.model_dir.lower():
        Model = DistilBertForSequenceClassification
    elif "roberta" in args.model_dir.lower():
        Model = RobertaForSequenceClassification
    elif "bert" in args.model_dir.lower():
        Model = BertForSequenceClassification
    config = AutoConfig.from_pretrained(args.model_dir, num_labels=args.num_labels)
    #config.bcos = args.bcos
    #config.b = args.b

    config.output_attentions = True
    config.num_labels = args.num_labels
    #print(config)
    model = Model.load_from_pretrained(args.model_dir, config=config, output_attentions=True)
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # Load a dataset from HuggingFace datasets library
    dataset = load_dataset(args.dataset_name)
    test_dataset = dataset[args.split]
    if "val" in dataset or args.split_ratio is None:
        test_dataset = Subset(test_dataset, range(len(test_dataset)))
    else:
        test_dataset = split_dataset(test_dataset, args.split_ratio)

    test_dataset = test_dataset[:] if args.num_examples == -1 else test_dataset[:args.num_examples]

    # add column index to the dataset
    test_dataset['index'] = list(range(len(test_dataset['text'])))

    # Initialize the explainer
    all_methods = EXPLANATION_METHODS.keys()
    if args.methods:
        attribution_methods = args.methods.replace(' ', '').split(',')   
    else:
        attribution_methods = all_methods  # Use all methods if none specified


    # Create output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for method in attribution_methods:
        print(f"\nRunning {method} explainer...")
        if EXPLANATION_METHODS[method] == BcosExplainer:
            explainer = BcosExplainer(model, tokenizer, args.relative)
        elif EXPLANATION_METHODS[method] == ShapleyValueExplainer:
            explainer = ShapleyValueExplainer(model, tokenizer, method, args.baseline, args.shap_n_samples)
        # for GradientNPropabationExplainer, we need to specify the method
        elif EXPLANATION_METHODS[method] == GradientNPropabationExplainer:
            explainer = EXPLANATION_METHODS[method](model, tokenizer, method, args.baseline)
        else:
            explainer = EXPLANATION_METHODS[method](model, tokenizer) 

        # can only explain the label class to reduce the computation time
        #class_labels = [dataset['label']]
        #explanation_results = explainer.explain_dataset(dataset, num_classes=args.num_labels, class_labels=class_labels, batch_size=args.batch_size, max_length=args.max_length)

        explanation_results = explainer.explain_dataset(test_dataset, num_classes=args.num_labels, batch_size=args.batch_size, max_length=args.max_length, only_predicted_classes=args.only_predicted_classes)
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
    parser.add_argument('--baseline', type=str, default='pad', help='Baseline for the attribution methods, select from zero, mask, pad')    
    parser.add_argument('--num_examples', type=int, default=1000, help='Number of examples to process (-1 for all)')
    parser.add_argument('--methods', type=str, default=None, help='List of attribution methods to use separated by commas')
    parser.add_argument('--output_dir', type=str, default='baseline_saliency_results/all_methods_1000_examples_512', help='Directory to save the output files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--shap_n_samples', type=int, default=25, help='Number of samples for Shapley Value Sampling')
    parser.add_argument('--only_predicted_classes', action='store_true', help='Only explain the predicted class')
    parser.add_argument('--relative', action='store_true', help='explain relative logits')

    args = parser.parse_args()
    main(args)

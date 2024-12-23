from saliency_utils.Explainer import BcosExplainer, AttentionExplainer, GradientNPropabationExplainer, OcclusionExplainer, ShapleyValueExplainer, LimeExplainer
from saliency_utils.utils import set_random_seed, split_dataset
from saliency_utils.human_agreement_utils import compute_human_agreement
import argparse
import torch
from transformers import AutoTokenizer, AutoConfig
from bcos_lm.models.new_modeling_bert import BertForSequenceClassification
from bcos_lm.models.new_modeling_roberta import RobertaForSequenceClassification
from bcos_lm.models.new_modeling_distilbert import DistilBertForSequenceClassification
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
    #args.num_examples = int(args.num_examples) if args.num_examples else None
    args.seed = int(args.seed) if args.seed else None
    args.shap_n_samples = int(args.shap_n_samples) if args.shap_n_samples else None
    percentages = [float(percentage) for percentage in args.percentages.split(',')]


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
    dataset = load_dataset(args.dataset_name, split=args.split, trust_remote_code=True)
    if args.num_examples > 0:
        dataset = dataset[:args.num_examples]
    # change 'review' to 'text' for the dataset
    dataset = {"text": dataset['review'], "label": dataset['label'], "evidences": dataset['evidences']}

    # add column index to the dataset
    dataset['index'] = list(range(len(dataset['text'])))
    evidences = dataset['evidences']

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
        output_file = os.path.join(args.output_dir, f'{method}_explanations.json')
        if os.path.exists(output_file):
            print(f"\nAttribution results for {method} already exist, skipping...")
            result = json.load(open(output_file, 'r'))
        else:
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

            explanation_results = explainer.explain_dataset(dataset, num_classes=args.num_labels, batch_size=args.batch_size, max_length=args.max_length, only_predicted_classes=args.only_predicted_classes)
            result = explanation_results

            # Save the results to a JSON file
            output_file = os.path.join(args.output_dir, f'{method}_explanations.json')
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"\nAttribution results saved to {output_file}")

        human_agreement_results = {}
        # compute human agreement
        for aggregation_method, attribution_results in result.items():
            print(f"\nComputing human agreement for {method} with {aggregation_method} aggregation...")
            assert len(attribution_results) == len(evidences)
            # select correctly predicted examples
            correct_indices = [i for i, e in enumerate(attribution_results) if e[0]['true_label'] == e[0]['predicted_class']]
            correct_attribution_results = [attribution_results[i][0] if len(attribution_results[i]) == 1 else attribution_results[i][dataset['label'][i]] for i in correct_indices]
            correct_evidences = [evidences[i] for i in correct_indices]
            print("Accuracy: ", len(correct_attribution_results)/len(attribution_results))
            # generate label_masks
            label_masks = []
            for attribution_result, evidence in zip(correct_attribution_results, correct_evidences):
                all_input_ids = tokenizer.encode(dataset['text'][attribution_result['index']], add_special_tokens=False)
                label_mask = [0 for _ in range(len(all_input_ids))]
                
                for evidence_span in evidence:
                    curr_idx = 0
                    span_ids = tokenizer.encode(evidence_span, add_special_tokens=False)
                    span_ids_with_prefix = tokenizer.encode(' '+evidence_span, add_special_tokens=False)

                    # locate the span in the input_ids
                    while curr_idx < len(all_input_ids):
                        if curr_idx < len(all_input_ids) - len(span_ids) and all_input_ids[curr_idx:curr_idx+len(span_ids)] == span_ids:
                            label_mask[curr_idx:curr_idx+len(span_ids)] = [1 for _ in span_ids]
                            
                            break
                        elif curr_idx < len(all_input_ids) - len(span_ids_with_prefix) and all_input_ids[curr_idx:curr_idx+len(span_ids_with_prefix)] == span_ids_with_prefix:
                            label_mask[curr_idx:curr_idx+len(span_ids_with_prefix)] = [1 for _ in span_ids_with_prefix]
                            
                            break
                        curr_idx += 1
                    
                # get real input ids and label masks
                if attribution_result['attribution'][0][0] == '[CLS]' or attribution_result['attribution'][0][0] == '<s>':
                    real_length = len([attr[0] for attr in attribution_result['attribution']]) - 2
                else:
                    real_length = len([attr[0] for attr in attribution_result['attribution']])
                label_masks.append(label_mask[:real_length])

            # compute human agreement
            auprc_results = []
            for attribution_result, label_mask in zip(correct_attribution_results, label_masks):
                attribution_scores = [attr[1] for attr in attribution_result['attribution']]
                input_ids = torch.tensor(tokenizer.convert_tokens_to_ids([attr[0] for attr in attribution_result['attribution']])).unsqueeze(0)
                attention_mask = torch.ones_like(input_ids)
                auprc_list = compute_human_agreement([attribution_scores], [label_mask], input_ids, attention_mask, percentages)
                auprc_results.extend(auprc_list)
            print(f"Mean AUPRC: {np.mean(auprc_results)}")
            human_agreement_results[aggregation_method] = {"auprc": np.mean(auprc_results), "accuracy":len(correct_attribution_results)/len(attribution_results), "auprc_list": auprc_results}
        # Save the human agreement results to a JSON file
        output_file = os.path.join(args.output_dir, f'{method}_human_agreement_results.json')
        with open(output_file, 'w') as f:
            json.dump(human_agreement_results, f, indent=4)

            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT Attribution with Captum')

    parser.add_argument('--dataset_name', type=str, default='eraser-benchmark/movie_rationales', help='Name of the HuggingFace dataset to use') #fancyzhx/ag_news, stanfordnlp/imdb
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, test)')
    #parser.add_argument('--split_ratio', type=float, default=0.5, help='Split ratio for test dataset')
    parser.add_argument('--model_dir', type=str, default='models/bert_base_imdb_512', help='Name of the pre-trained model')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels in the classification')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length for tokenization')
    parser.add_argument('--baseline', type=str, default='pad', help='Baseline for the attribution methods, select from zero, mask, pad')    
    parser.add_argument('--num_examples', type=int, default=-1, help='Number of examples to process (-1 for all)')
    parser.add_argument('--methods', type=str, default=None, help='List of attribution methods to use separated by commas')
    parser.add_argument('--output_dir', type=str, default='baseline_saliency_results/all_methods_1000_examples_512', help='Directory to save the output files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--percentages', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9', help='Comma-separated list of percentages for selecting rationales') 
    parser.add_argument('--shap_n_samples', type=int, default=25, help='Number of samples for Shapley Value Sampling')
    parser.add_argument('--only_predicted_classes', action='store_true', help='Only explain the predicted class')
    #parser.add_argument('--bcos', action='store_true', help='Use Bcos model')
    #parser.add_argument('--b', type=float, default=2.0, help='Bcos parameter')
    parser.add_argument('--relative', action='store_true', help='explain relative logits')

    args = parser.parse_args()
    main(args)

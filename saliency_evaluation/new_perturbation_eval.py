import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from bcos_lm.models.new_new_modeling_bert import BertForSequenceClassification
from bcos_lm.models.new_new_modeling_roberta import RobertaForSequenceClassification
from bcos_lm.models.new_new_modeling_distilbert import DistilBertForSequenceClassification
from saliency_utils.perturbation_utils import select_rationales, compute_comprehensiveness, compute_sufficiency, compute_perturbation_auc
from argparse import ArgumentParser
import json
import random
import numpy as np
from tqdm import tqdm
import os

def batch_loader(data, batch_size):
    # yield batches of data; if the last batch is smaller than batch_size, return the smaller batch
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def is_embedding_attribution(method):
    # check if the prediction is attributed to the embeddings (token, position, token_type)
    if "_mean" in method or "_L2" in method:
        return True
    return False

def main(args):

    # convert strings to numbers
    args.num_labels = int(args.num_labels) if args.num_labels else None
    args.batch_size = int(args.batch_size) if args.batch_size else None
    args.max_length = int(args.max_length) if args.max_length else None
    args.num_examples = int(args.num_examples) if args.num_examples else None
    args.seed = int(args.seed) if args.seed else None
    #args.embedding_attributions = args.embedding_attributions if args.embedding_attributions != 'None' else None

    # Set random seed for reproducibility
    def set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_random_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    # Load tokenizer and model
    if "distilbert" in args.model_dir.lower():
        Model = DistilBertForSequenceClassification
    elif "roberta" in args.model_dir.lower():
        Model = RobertaForSequenceClassification
    elif "bert" in args.model_dir.lower():
        Model = BertForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    config = AutoConfig.from_pretrained(args.model_dir, num_labels=args.num_labels)
    config.output_attentions = True
    config.num_labels = args.num_labels
    #print(config)
    model = Model.load_from_pretrained(args.model_dir, config=config).to(device)
    model.eval()
    if args.mask_type == "mask":
        mask_token_id = tokenizer.mask_token_id
    elif args.mask_type == "pad":
        mask_token_id = tokenizer.pad_token_id
    else:
        raise ValueError("Invalid mask type. Choose from 'mask' or 'pad'.")
    # load data

    # find all files under the explanation_dir
    explanation_paths = [os.path.join(args.explanation_dir, f) for f in os.listdir(args.explanation_dir) if os.path.isfile(os.path.join(args.explanation_dir, f)) and "perturbation" not in f and f.endswith('.json')]
    for explanation_path in explanation_paths:
        with open(explanation_path) as f:
            saliency_data = json.load(f)
        print(f"Loaded saliency data from {explanation_path}")

        methods = saliency_data.keys()
        percentages = [float(percentage) for percentage in args.percentages.split(',')]
        perturbation_results = {method: {} for method in methods}

        for method in methods:
            print(f"Method: {method}")
            # convert text, target_class, attribution to dataloader
            data = saliency_data[method]
            # filter out instances where the predicted class is not the target class
            correctly_predicted_data = [expl for instance in data for expl in instance if expl['predicted_class']==expl['target_class']]
            #print(len(data), len(correctly_predicted_data))
            assert len(data) == len(correctly_predicted_data), "Some instances have different predicted and target classes"
            if args.num_examples > 0:
                correctly_predicted_data = correctly_predicted_data[:args.num_examples]
            
            #dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
            dataloader = batch_loader(correctly_predicted_data, args.batch_size)
            perturbation_results[method] = {str(percentage): {"comprehensiveness_list": [], "sufficiency_list": []} for percentage in percentages}
            # set different percentage of rationales
    
            for idx, batch in tqdm(enumerate(dataloader)):
                #texts = [x['text'] for x in batch]
                predicted_classes = torch.tensor([x['predicted_class'] for x in batch]).to(device)
                input_tokens = [[expl[0] for expl in attr] for attr in [x['attribution'] for x in batch]]
                input_ids = torch.ones((len(batch), args.max_length), dtype=torch.long) * tokenizer.pad_token_id
                attention_mask = torch.zeros((len(batch), args.max_length), dtype=torch.long)
                for i, tokens in enumerate(input_tokens):
                    input_ids[i, :len(tokens)] = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))
                    attention_mask[i, :len(tokens)] = 1
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                #encodings = tokenizer(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt')
                #input_ids = encodings['input_ids'].to(device)        # Shape: [batch_size, seq_length]
                #attention_mask = encodings['attention_mask'].to(device)  # Shape: [batch_size, seq_length]
                # compute original probs
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                orig_logits = outputs.logits
                orig_probs = torch.softmax(orig_logits, dim=-1)
                # gather the predicted class and the probabilities for these classes
                predicted_ids = torch.argmax(orig_probs, dim=1)
                #assert torch.equal(predicted_ids, predicted_classes), "Predicted class does not match the target class"
                orig_probs = orig_probs.gather(1, predicted_ids.unsqueeze(1)).squeeze(1)  # Shape: [batch_size]
                
                #predicted_ids = predicted_classes
                #orig_probs = torch.tensor([x['predicted_class_confidence'] for x in batch]).to(device)
                
                # get the attributions
                attributions = [[expl[1] for expl in attr] for attr in [x['attribution'] for x in batch]]

                for percentage in percentages:
                    rationale_mask = select_rationales(attributions, input_ids, attention_mask, percentage)
                    comprehensiveness = compute_comprehensiveness(model, input_ids, attention_mask, rationale_mask, predicted_ids, orig_probs, mask_token_id)
                    sufficiency = compute_sufficiency(model, input_ids, attention_mask, rationale_mask, predicted_ids, orig_probs, mask_token_id)
                    perturbation_results[method][str(percentage)]["comprehensiveness_list"].extend(comprehensiveness.cpu().numpy().tolist())
                    perturbation_results[method][str(percentage)]["sufficiency_list"].extend(sufficiency.cpu().numpy().tolist())
            for percentage in percentages:  
                perturbation_results[method][str(percentage)]["comprehensiveness_score"] = np.mean(perturbation_results[method][str(percentage)]["comprehensiveness_list"])   
                perturbation_results[method][str(percentage)]["sufficiency_score"] = np.mean(perturbation_results[method][str(percentage)]["sufficiency_list"]) 
            # compute AUC
            comprehensiveness_scores = [perturbation_results[method][str(percentage)]["comprehensiveness_score"] for percentage in percentages]
            sufficiency_scores = [perturbation_results[method][str(percentage)]["sufficiency_score"] for percentage in percentages]
            comprehensiveness_auc = compute_perturbation_auc(percentages, comprehensiveness_scores)
            sufficiency_auc = compute_perturbation_auc(percentages, sufficiency_scores)
            perturbation_results[method]["comprehensiveness_auc"] = comprehensiveness_auc
            perturbation_results[method]["sufficiency_auc"] = sufficiency_auc
            print(f"Comprehensiveness AUC: {comprehensiveness_auc}")
            print(f"Sufficiency AUC: {sufficiency_auc}")
        output_path = explanation_path.replace('explanations.json', 'perturbation_results.json')
        with open(output_path, 'w') as f:
            json.dump(perturbation_results, f, indent=4)
        print(f"Results saved to {output_path}")

if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluate the faithfulness for rationales using perturbation-based methods.')

    parser.add_argument('--explanation_dir', type=str, default='baseline_saliency_results/all_methods_1000_examples_512', help='Path to the saliency data')
    parser.add_argument('--model_dir', type=str, default='models/bert_base_imdb_512', help='Name of the pre-trained model')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of classes in the dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length for tokenization')
    parser.add_argument('--num_examples', type=int, default=-1, help='Number of examples to process (-1 for all)')
    #parser.add_argument('--methods', type=str, default='', help='Comma-separated list of attribution methods to use')
    #parser.add_argument('--embedding_attributions', nargs='+', default=[], help='List of embeddings to attribute the prediction to')
    parser.add_argument('--mask_type', type=str, default='mask', help='Type of token to mask for perturbation')
    parser.add_argument('--percentages', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9', help='Comma-separated list of percentages for selecting rationales')
    #parser.add_argument('--output_path', type=str, default='baseline_saliency_results/all_methods_1000_examples_512/Attention_perturbation_results.json', help='Directory to save the output files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    #parser.add_argument('--bcos', action='store_true', help='Use BCOS model')
    #parser.add_argument('--b', type=float, default=2.0, help='BCOS parameter')

    args = parser.parse_args()
    main(args)
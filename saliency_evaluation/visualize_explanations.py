import torch
import numpy as np
import json
from datasets import load_dataset
from saliency_utils.perturbation_utils import select_rationales
from transformers import AutoTokenizer, AutoConfig

def visualize_explanation(attribution_result, percentage, tokenizer):
    tokens = [attr[0] for attr in attribution_result['attribution']]
    if tokens[0] != ['CLS']:
        tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    attribution_scores = [attr[1] for attr in attribution_result['attribution']]
    rationale_mask = select_rationales([attribution_scores], input_ids, attention_mask, percentage)
    rationale_mask = rationale_mask.cpu().numpy().tolist()[0]
    # compute longest 1 sequence in the rationale mask 
    max_len = 0
    curr_len = 0
    for mask in rationale_mask[1:-1]:
        if mask == 1:
            curr_len += 1
        else:
            max_len = max(max_len, curr_len)
            curr_len = 0
    
    print(f"True label is: {attribution_result['true_label']}, Predicted label is: {attribution_result['predicted_class']}.")
    # print the sentence with [token] when the token is selected as rationale
    print(f"Rationale with percentage {percentage}:")
    for i, (token, mask) in enumerate(zip(tokens, rationale_mask)):
        if token == '[CLS]' or token == '[SEP]':
            continue
        if mask:
            print("[[{}]]".format(token), end=" ")
        else:
            print(token, end=" ")
    print("\n")
    print("Longest 1 sequence in the rationale mask is: ", max_len)
    print("\n")

def visualize_human_explanation(attribution_result, evidence, orig_text, tokenizer):
    all_input_ids = tokenizer.encode(orig_text, add_special_tokens=False)
    tokens = [attr[0] for attr in attribution_result['attribution']]
    label_mask = [0 for _ in range(len(all_input_ids))]
    curr_idx = 0

    for evidence_span in evidence:
        curr_idx = 0
        span_ids = tokenizer.encode(evidence_span, add_special_tokens=False)

        # locate the span in the input_ids
        while curr_idx < len(all_input_ids) - len(span_ids):
            if all_input_ids[curr_idx:curr_idx+len(span_ids)] == span_ids:
                label_mask[curr_idx:curr_idx+len(span_ids)] = [1 for _ in span_ids]
                break
            curr_idx += 1
    # get real input ids and label masks
    if attribution_result['attribution'][0][0] == '[CLS]':
        real_length = len(tokens) - 2
        tokens = tokens[1:-1]
    else:
        real_length = len(tokens)
    label_mask = label_mask[:real_length]
    # compute longest 1 sequence in the label mask
    max_len = 0
    curr_len = 0
    for mask in label_mask:
        if mask == 1:
            curr_len += 1
        else:
            max_len = max(max_len, curr_len)
            curr_len = 0

    print(f"True label is: {attribution_result['true_label']}.")
    print(f"Human Rationale:")
    for i, (token, mask) in enumerate(zip(tokens, label_mask)):
        if mask==1:
            print("[[{}]]".format(token), end=" ")
        else:
            print(token, end=" ")
    print("\n")
    print("Longest 1 sequence in the label mask is: ", max_len)
    print("\n")
    return sum(label_mask) / len(label_mask)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    explanation_path = "bcos_results/pointing_game_agnews_bce_no_embedding_norm_gelu_new_implementation_train_100/Bcos_explanations.json"
    method = "Bcos_absolute_ixg_mean"
    percentage = 0.2
    idx = 10

    with open(explanation_path, 'r') as f:
        attribution_results = json.load(f)[method]

    dataset = load_dataset("eraser-benchmark/movie_rationales", split="test")
    # change 'review' to 'text' for the dataset
    dataset = {"text": dataset['review'], "label": dataset['label'], "evidences": dataset['evidences']}
    # add column index to the dataset
    dataset['index'] = list(range(len(dataset['text'])))
    evidences = dataset['evidences']

    # select correctly predicted examples
    correct_indices = [i for i, e in enumerate(attribution_results) if e[0]['true_label'] == e[0]['predicted_class']]
    correct_attribution_results = [attribution_results[i][0] if len(attribution_results[i]) == 1 else attribution_results[i][dataset['label'][i]] for i in correct_indices]
    correct_evidences = [evidences[i] for i in correct_indices]
    correct_orig_texts = [dataset['text'][i] for i in correct_indices]

    # visualize human evaluation
    human_percentage = visualize_human_explanation(correct_attribution_results[idx], correct_evidences[idx], correct_orig_texts[idx], tokenizer)

    # visualize model explanation
    visualize_explanation(correct_attribution_results[idx], human_percentage, tokenizer)

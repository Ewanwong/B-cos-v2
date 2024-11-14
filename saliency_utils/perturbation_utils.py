import torch
import numpy as np
from sklearn.metrics import auc

def select_rationales(attribution_scores, input_ids, attention_mask, percentage):
    """
    Select top percentage of tokens as rationales based on attribution scores for a batch.

    Args:
        attribution_scores List[List]: Attribution scores for each token. 
                                           Shape: [batch_size, seq_length]
        percentage (float): Percentage of tokens to select as rationales (between 0 and 1).

    Returns:
        rationale_mask (torch.Tensor): Boolean mask indicating selected rationales.
                                       Shape: [batch_size, seq_length]
    """
    #TODO: consider the attribution that does not contain [CLS] and [SEP] tokens

    batch_size, seq_length = input_ids.size()
    rationale_mask = torch.zeros_like(input_ids, dtype=torch.bool).to(input_ids.device)
    if percentage == 0.0:
        return rationale_mask
    if percentage == 1.0:
        return torch.ones_like(input_ids, dtype=torch.bool).to(input_ids.device)        
    # compute the real length of each input
    real_length = torch.sum(attention_mask, dim=1)

    # minus 2 for [CLS] and [SEP] tokens
    real_length -= 2

    # compute the number of tokens to select for each example
    k = (real_length * percentage).clamp(min=1).long()  # Ensure at least one token is selected

    # For each example in the batch, select top-k tokens
    for i in range(batch_size):
        topk = k if isinstance(k, int) else k[i]
        #print(real_length[i], len(attribution_scores[i]))
        if real_length[i] == len(attribution_scores[i]):
            # select the top k tokens from the list of attribution scores
            # if the length of the attribution score is the same as the input length, that means the attribution score does not contain [CLS] and [SEP] tokens
            topk_indices = np.argsort(attribution_scores[i])[-topk:][::-1]
        elif real_length[i] == len(attribution_scores[i]) - 2:           
            topk_indices = np.argsort(attribution_scores[i][1:-1])[-topk:][::-1] # exclude [CLS] and [SEP] tokens in the attribution score
        else:
            raise ValueError("The length of the attribution score does not match the input length")
        # add 1 to skip [CLS] token
        topk_indices += 1
        rationale_mask[i, topk_indices.copy()] = True # make sure the selected token is not [CLS] or [SEP]

    return rationale_mask

def compute_comprehensiveness(model, input_ids, attention_mask, rationale_mask, predicted_ids, orig_probs, mask_token_id):
    """
    Compute the comprehensiveness score by masking out the rationales for a batch.

    Args:
        model: The BERT model.
        input_ids (torch.Tensor): Input token IDs. Shape: [batch_size, seq_length]
        attention_mask (torch.Tensor): Attention mask. Shape: [batch_size, seq_length]
        rationale_mask (torch.Tensor): Boolean mask for rationales. Shape: [batch_size, seq_length]

    Returns:
        comprehensiveness (torch.Tensor): Comprehensiveness scores for each example. Shape: [batch_size]
    """
    model.eval()
    with torch.no_grad():
        if orig_probs is None or predicted_ids is None:
            # Original prediction
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            orig_logits = outputs.logits
            orig_probs = torch.softmax(orig_logits, dim=-1)
            # gather the predicted class and the probabilities for these classes
            predicted_ids = torch.argmax(orig_probs, dim=1)
            orig_probs = orig_probs.gather(1, predicted_ids.unsqueeze(1)).squeeze(1)  # Shape: [batch_size]


        # Mask out rationales
        masked_input_ids = input_ids.clone()
        masked_input_ids[rationale_mask] = mask_token_id  # Mask token IDs

        outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask)
        masked_logits = outputs.logits
        masked_probs = torch.softmax(masked_logits, dim=-1)
        masked_probs = masked_probs.gather(1, predicted_ids.unsqueeze(1)).squeeze(1)  # Shape: [batch_size]

        comprehensiveness = orig_probs - masked_probs

    return comprehensiveness

def compute_sufficiency(model, input_ids, attention_mask, rationale_mask, predicted_ids, orig_probs, mask_token_id):
    """
    Compute the sufficiency score by keeping only the rationales for a batch.

    Args:
        model: The BERT model.
        input_ids (torch.Tensor): Input token IDs. Shape: [batch_size, seq_length]
        attention_mask (torch.Tensor): Attention mask. Shape: [batch_size, seq_length]
        rationale_mask (torch.Tensor): Boolean mask for rationales. Shape: [batch_size, seq_length]
        label_ids (torch.Tensor): The target label indices. Shape: [batch_size]

    Returns:
        sufficiency (torch.Tensor): Sufficiency scores for each example. Shape: [batch_size]
    """
    model.eval()
    with torch.no_grad():
        if orig_probs is None or predicted_ids is None:
            # Original prediction
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            orig_logits = outputs.logits
            orig_probs = torch.softmax(orig_logits, dim=-1)
            # gather the predicted class and the probabilities for these classes
            predicted_ids = torch.argmax(orig_probs, dim=1)
            orig_probs = orig_probs.gather(1, predicted_ids.unsqueeze(1)).squeeze(1)  # Shape: [batch_size]

        # Keep only rationales
        suff_input_ids = input_ids.clone()
        suff_input_ids[~rationale_mask] = mask_token_id  # Mask non-rationales

        outputs = model(input_ids=suff_input_ids, attention_mask=attention_mask)
        suff_logits = outputs.logits
        suff_probs = torch.softmax(suff_logits, dim=-1)
        suff_probs = suff_probs.gather(1, predicted_ids.unsqueeze(1)).squeeze(1)  # Shape: [batch_size]

        sufficiency = orig_probs - suff_probs

    return sufficiency

def compute_perturbation_auc(percentages, scores):
    """
    Compute the AUC for the perturbation scores at different percentages.

    Args:
        percentages (List[float]): List of percentages.
        scores (List[float]): List of scores corresponding to the percentages.

    Returns:
        auc_score (float): The AUC score.
    """
    auc_score = auc(percentages, scores)
    return auc_score
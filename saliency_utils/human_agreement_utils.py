import torch
import numpy as np
from sklearn.metrics import auc
from saliency_utils.perturbation_utils import select_rationales

def compute_precision_recall(predicted_mask, label_mask):
    # predicted mask and label mask are both 0/1 list, compute precision and recall
    assert len(predicted_mask) == len(label_mask) or len(predicted_mask) == len(label_mask) + 2, "The length of the predicted mask and the label mask should be the same"
    if len(predicted_mask) == len(label_mask) + 2:
        predicted_mask = predicted_mask[1:-1]
    predicted_mask = np.array(predicted_mask)
    label_mask = np.array(label_mask)
    tp = np.sum(predicted_mask * label_mask)
    fp = np.sum(predicted_mask * (1 - label_mask))
    fn = np.sum((1 - predicted_mask) * label_mask)
    precision = tp / (tp + fp) if tp + fp > 0 else 0    
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return precision, recall

def compute_auprc(predicted_masks, label_mask):
    precisions = []
    recalls = []
    for predicted_mask in predicted_masks:
        precision, recall = compute_precision_recall(predicted_mask, label_mask)
        precisions.append(precision)
        recalls.append(recall)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]
    auprc = auc(recalls, precisions)
    return auprc

def compute_human_agreement(attribution_scores, label_masks, input_ids, attention_masks, percentages):
    
    # compute the rationales
    rationale_masks = []
    for i, attribution_score in enumerate(attribution_scores):
        masks = []
        for percentage in percentages:     
            mask = select_rationales([attribution_score], input_ids[i:i+1, :], attention_masks[i:i+1, :], percentage).cpu().numpy().tolist()     
            masks.append(mask[0])
        rationale_masks.append(masks)
    # compute the human agreement
    auprc_list = []
    for predicted_masks, label_mask in zip(rationale_masks, label_masks):
        auprc = compute_auprc(predicted_masks, label_mask)
        auprc_list.append(auprc)
    return auprc_list




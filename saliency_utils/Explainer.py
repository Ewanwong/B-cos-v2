import argparse
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from captum.attr import Saliency, DeepLift, GuidedBackprop, InputXGradient, IntegratedGradients, Occlusion, ShapleyValueSampling 
from saliency_utils.lime_utils import explain
from datasets import load_dataset
import numpy as np
import json
import os
import random
from tqdm import tqdm
from saliency_utils.utils import batch_loader

#TODO: add explain hybrid documents for datasets

class BertEmbeddingModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(BertEmbeddingModelWrapper, self).__init__()
        self.model = model

    def forward(self, token_embeddings, position_embeddings, token_type_embeddings, attention_mask=None):
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        extended_attention_mask = self.model.get_extended_attention_mask(
            attention_mask, embeddings.shape[:2], embeddings.device
        )
        embeddings = self.model.bert.embeddings.LayerNorm(embeddings)
        embeddings = self.model.bert.embeddings.dropout(embeddings)
        #head_mask = [None] * self.model.config.num_hidden_layers
        head_mask = self.model.get_head_mask(None, self.model.config.num_hidden_layers) 
        encoder_outputs = self.model.bert.encoder(
            embeddings,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.model.bert.pooler(sequence_output) if self.model.bert.pooler is not None else None
        pooled_ouput = self.model.dropout(pooled_output)
        logits = self.model.classifier(pooled_output)
        return logits

class BertModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(BertModelWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None):       
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
class BertProbabilityModelWrapper(torch.nn.Module):
    def __init__(self, model, tokenizer, batch_size=16):
        super(BertProbabilityModelWrapper, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.bsz = batch_size

    def forward(self, texts):
        # process inputs by batch
        bsz = self.bsz
        if len(texts) > bsz:
            batch_size = bsz
            n_batches = len(texts) // batch_size
            if len(texts) % batch_size != 0:
                n_batches += 1
            probabilities = []
            for i in range(n_batches):
                batch = texts[i*batch_size:(i+1)*batch_size]
                batch_prob = self._predict_batch(batch)
                probabilities.append(batch_prob)
            probabilities = np.concatenate(probabilities, axis=0)
        else:
            probabilities = self._predict_batch(texts)
        return probabilities
    
    def _predict_batch(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.detach().cpu().numpy()
    
class BaseExplainer:
    def explain(self):
        raise NotImplementedError
    
class AttentionExplainer(BaseExplainer):
    def __init__(self, model, tokenizer):
        # attention explainer can only explain the predicted classes
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = model.device

    def _explain(self, input_ids, attention_mask, example_indices=None, labels=None, num_classes=None, class_labels=None):

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        attentions = outputs.attentions
        predicted_classes = torch.argmax(outputs.logits, dim=1).detach().cpu().numpy().tolist()

        # confidence for each class
        confidences = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().cpu().numpy().tolist()

        # Stack attentions over layers
        all_attentions = torch.stack(attentions)
        # Get sequence length and batch size
        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        # Expand attention mask to match attention shapes
        # Shape: (batch_size, 1, 1, seq_len)
        attention_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)

        # Create a mask for attention weights
        # Shape: (batch_size, 1, seq_len, seq_len)
        attention_mask_matrix = attention_mask_expanded * attention_mask_expanded.transpose(-1, -2)

        # Mask out padding tokens in attention weights
        # We set the attention weights corresponding to padding tokens to zero
        all_attentions = all_attentions * attention_mask_matrix.unsqueeze(0)

        # Normalize the attention weights so that they sum to 1 over the real tokens
        # Sum over the last dimension (seq_len)
        attn_weights_sum = all_attentions.sum(dim=-1, keepdim=True) + 1e-9  # Add epsilon to avoid division by zero
        all_attentions = all_attentions / attn_weights_sum

        # Convert input IDs back to tokens
        tokens_batch = [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]

        # Average Attention
        # Average over heads
        avg_attn_heads = all_attentions.mean(dim=2)  # Shape: (num_layers, batch_size, seq_len, seq_len)
        # Average over layers
        avg_attn = avg_attn_heads.mean(dim=0)  # Shape: (batch_size, seq_len, seq_len)

        # Attention Rollout
        rollout = torch.eye(seq_len).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)  # Shape: (batch_size, seq_len, seq_len)
        for attn in avg_attn_heads:
            attn = attn + torch.eye(seq_len).unsqueeze(0).to(self.device)  # Add identity for self-connections
            attn = attn / attn.sum(dim=-1, keepdim=True)  # Normalize rows
            rollout = torch.bmm(rollout, attn)  # Batch matrix multiplication

        # Attention Flow
        # Take maximum over heads
        attn_per_layer_max = all_attentions.max(dim=2)[0]  # Shape: (num_layers, batch_size, seq_len, seq_len)
        # Initialize cumulative attention starting from [CLS]
        cumulative_attn = torch.zeros(batch_size, seq_len).to(self.device)
        cumulative_attn[:, 0] = 1.0  # [CLS] token index is 0
        for attn in attn_per_layer_max:
            # attn shape: (batch_size, seq_len, seq_len)
            # cumulative_attn shape: (batch_size, seq_len)
            # Compute maximum attention flow to each token
            cumulative_attn = torch.max(cumulative_attn.unsqueeze(-1) * attn, dim=1)[0]
        flow_cls_attn = cumulative_attn  # Shape: (batch_size, seq_len)

        # Extract attention from [CLS] token
        avg_cls_attn = avg_attn[:, 0, :]  # Shape: (batch_size, seq_len)
        rollout_cls_attn = rollout[:, 0, :]  # Shape: (batch_size, seq_len)

        all_raw_attention_explanations = []
        all_attention_rollout_explanations = []
        all_attention_flow_explanations = []
        # For each example in the batch, print the attention scores
        for i in range(batch_size):
            each_raw_attention_explanations = []
            each_attention_rollout_explanations = []
            each_attention_flow_explanations = []
            tokens = tokens_batch[i]                
            predicted_class = predicted_classes[i]
            if labels is not None:
                true_label = labels[i]
            else:
                true_label = None 
            valid_len = attention_mask[i].sum().item()  # Number of real tokens
            raw_attention_attribution = avg_cls_attn[i][:int(valid_len)].cpu().numpy()
            rollout_attribution = rollout_cls_attn[i][:int(valid_len)].cpu().numpy()
            flow_attribution = flow_cls_attn[i][:int(valid_len)].cpu().numpy()
            if num_classes is not None:
                class_indices = list(range(num_classes))
            else:
                class_indices = [predicted_class]
            if class_labels is not None:
                class_indices = [class_label[i] for class_label in class_labels]
            for class_idx in class_indices:
            # skip all padding tokens
                raw_attention_result = {
                    'index': example_indices[i],
                    'text': self.tokenizer.decode(input_ids[i], skip_special_tokens=True),
                    #'tokens': tokens[:int(valid_len)],
                    'true_label': true_label,
                    'predicted_class': predicted_class,
                    'predicted_class_confidence': confidences[i][predicted_class],
                    'target_class': class_idx,
                    'target_class_confidence': confidences[i][class_idx],
                    'method': 'raw_attention',
                    'attribution': list(zip(tokens[:int(valid_len)], raw_attention_attribution.tolist())),
                }
                rollout_result = {
                    'index': example_indices[i],
                    'text': self.tokenizer.decode(input_ids[i], skip_special_tokens=True),
                    #'tokens': tokens[:int(valid_len)],
                    'true_label': true_label,
                    'predicted_class': predicted_class,
                    'predicted_class_confidence': confidences[i][predicted_class],
                    'target_class': class_idx,
                    'target_class_confidence': confidences[i][class_idx],
                    'method': 'attention_rollout',
                    'attribution': list(zip(tokens[:int(valid_len)], rollout_attribution.tolist())),
                }

                flow_result = {
                    'index': example_indices[i],
                    'text': self.tokenizer.decode(input_ids[i], skip_special_tokens=True),
                    #'tokens': tokens[:int(valid_len)],
                    'true_label': true_label,
                    'predicted_class': predicted_class,
                    'predicted_class_confidence': confidences[i][predicted_class],
                    'target_class': class_idx,
                    'target_class_confidence': confidences[i][class_idx],
                    'method': 'attention_flow',
                    'attribution': list(zip(tokens[:int(valid_len)], flow_attribution.tolist())),
                }
                each_raw_attention_explanations.append(raw_attention_result)
                each_attention_rollout_explanations.append(rollout_result)
                each_attention_flow_explanations.append(flow_result)
            all_raw_attention_explanations.append(each_raw_attention_explanations)
            all_attention_rollout_explanations.append(each_attention_rollout_explanations)
            all_attention_flow_explanations.append(each_attention_flow_explanations)
        attention_explanations = {"raw_attention": all_raw_attention_explanations, "attention_rollout": all_attention_rollout_explanations, "attention_flow": all_attention_flow_explanations}
        return attention_explanations
        
    def explain(self, texts, example_indices, labels=None, num_classes=None, class_labels=None, max_length=512):

        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        attention_explanations = self._explain(input_ids=input_ids, attention_mask=attention_mask, example_indices=example_indices, labels=labels, num_classes=num_classes, class_labels=class_labels)
        return attention_explanations
    
    def explain_hybrid_documents(self, text1, text2, example_indices, labels=None, num_classes=None, class_labels=None, max_length=512):

        inputs = self.tokenizer(text1, text2, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        attention_explanations = self._explain(input_ids=input_ids, attention_mask=attention_mask, example_indices=example_indices, labels=labels, num_classes=num_classes, class_labels=class_labels)
        return attention_explanations
    
    def explain_dataset(self, dataset, num_classes=None, class_labels=None, batch_size=16, max_length=512):
        # classes_labels: [[class1 for ex1, class1 for ex2...], [class2 for ex1, class2 for ex2...]]
        if num_classes is not None or class_labels is not None:
            print("Attention explainer can only explain the predicted classes")
        data_loader = batch_loader(dataset, batch_size=batch_size, shuffle=False)
        raw_attention_explanations = []
        attention_rollout_explanations = []
        attention_flow_explanations = []
        class_labels_indexer = 0
        for batch in tqdm(data_loader):
            example_indices = batch['index']
            texts = batch['text']
            labels = batch['label']
            if class_labels is not None:
                batch_class_labels = [predicted_label[class_labels_indexer: class_labels_indexer+len(example_indices)] for predicted_label in class_labels]
                class_labels_indexer += len(example_indices)
            else:
                batch_class_labels = None
            attention_explanation = self.explain(texts=texts, example_indices=example_indices, labels=labels, num_classes=num_classes, class_labels=batch_class_labels, max_length=max_length)
            raw_attention_explanations.extend(attention_explanation['raw_attention'])
            attention_rollout_explanations.extend(attention_explanation['attention_rollout'])
            attention_flow_explanations.extend(attention_explanation['attention_flow'])
        attention_explanations = {"raw_attention": raw_attention_explanations, "attention_rollout": attention_rollout_explanations, "attention_flow": attention_flow_explanations}
        return attention_explanations
    
    def explain_hybrid_documents_dataset(self, dataset, num_classes=None, class_labels=None, batch_size=16, max_length=512):

        data_loader = batch_loader(dataset, batch_size=batch_size, shuffle=False)
        raw_attention_explanations = []
        attention_rollout_explanations = []
        attention_flow_explanations = []
        class_labels_indexer = 0
        for batch in tqdm(data_loader):
            example_indices = batch['index']
            texts1 = batch['text1']
            texts2 = batch['text2']
            labels = None
            if class_labels is not None:
                batch_class_labels = [predicted_label[class_labels_indexer: class_labels_indexer+len(example_indices)] for predicted_label in class_labels]
                class_labels_indexer += len(example_indices)
            else:
                batch_class_labels = None
            attention_explanation = self.explain_hybrid_documents(text1=texts1, text2=texts2, example_indices=example_indices, labels=labels, num_classes=num_classes, class_labels=batch_class_labels, max_length=max_length)
            raw_attention_explanations.extend(attention_explanation['raw_attention'])
            attention_rollout_explanations.extend(attention_explanation['attention_rollout'])
            attention_flow_explanations.extend(attention_explanation['attention_flow'])
        attention_explanations = {"raw_attention": raw_attention_explanations, "attention_rollout": attention_rollout_explanations, "attention_flow": attention_flow_explanations}
        return attention_explanations

class GradientNPropabationExplainer(BaseExplainer):
    def __init__(self, model, tokenizer, method='saliency'):
        self.model = BertEmbeddingModelWrapper(model)
        self.model.eval()
        self.model.to(model.device)
        self.tokenizer = tokenizer
        self.method = method
        if method == 'Saliency':
            self.explainer = Saliency(self.model)
        elif method == 'InputXGradient':
            self.explainer = InputXGradient(self.model)
        elif method == 'IntegratedGradients':
            self.explainer = IntegratedGradients(self.model)
        elif method == 'DeepLift':
            self.explainer = DeepLift(self.model)
        elif method == 'GuidedBackprop':
            self.explainer = GuidedBackprop(self.model)
        else:
            raise ValueError(f"Invalid method {method}")
        self.device = model.device

    def _explain(self, input_ids, attention_mask, position_ids=None, token_type_ids=None, example_indices=None, labels=None, num_classes=None, class_labels=None):

        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=self.device).unsqueeze(0).repeat(input_ids.size(0), 1)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        batch_size = input_ids.shape[0]


        # Extract embeddings
        token_embeddings = self.model.model.bert.embeddings.word_embeddings(input_ids)
        position_embeddings = self.model.model.bert.embeddings.position_embeddings(position_ids)
        token_type_embeddings = self.model.model.bert.embeddings.token_type_embeddings(token_type_ids)

        # Get the model's predictions
        with torch.no_grad():
            outputs = self.model(token_embeddings, position_embeddings, token_type_embeddings, attention_mask=attention_mask)
        predicted_classes = outputs.argmax(dim=-1).detach().cpu().numpy().tolist()
        # confidence for each class
        confidences = torch.nn.functional.softmax(outputs, dim=-1).detach().cpu().numpy().tolist()

        # Set requires_grad to True for embeddings we want to compute attributions for
        token_embeddings.requires_grad_()
        position_embeddings.requires_grad_()
        token_type_embeddings.requires_grad_()

        input_ids_cpu = input_ids.detach().cpu().numpy().tolist()
        all_explained_labels = []
        if class_labels is None:            
            # explain for all classes
            for class_idx in range(num_classes):
                class_labels = [class_idx] * batch_size
                all_explained_labels.append(class_labels)
        else:
            all_explained_labels=class_labels
        
        all_saliency_l2_results = [[] for _ in range(batch_size)]
        all_saliency_mean_results = [[] for _ in range(batch_size)]
        for explained_labels in all_explained_labels:
            attributions = self.explainer.attribute(
                inputs=(token_embeddings, position_embeddings, token_type_embeddings),
                target=explained_labels,
                additional_forward_args=(attention_mask,)
            )
            attributions_token, attributions_position, attributions_token_type = attributions


            for i in range(batch_size):

                tokens = self.tokenizer.convert_ids_to_tokens(input_ids_cpu[i])
                class_index = explained_labels[i]
                predicted_class = predicted_classes[i]
                if labels is not None:
                    true_label = labels[i]
                else:
                    true_label = None                    

                # Compute saliency metrics for each token
                saliency_token_l2 = torch.norm(attributions_token[i:i+1], dim=-1).detach().cpu().numpy()[0]
                saliency_token_mean = attributions_token[i:i+1].mean(dim=-1).detach().cpu().numpy()[0]
                saliency_position_l2 = torch.norm(attributions_position[i:i+1], dim=-1).detach().cpu().numpy()[0]
                saliency_position_mean = attributions_position[i:i+1].mean(dim=-1).detach().cpu().numpy()[0]
                saliency_token_type_l2 = torch.norm(attributions_token_type[i:i+1], dim=-1).detach().cpu().numpy()[0]
                saliency_token_type_mean = attributions_token_type[i:i+1].mean(dim=-1).detach().cpu().numpy()[0]
                # Collect results for the current example and class
                # skip padding tokens
                tokens = [token for token in tokens if token != self.tokenizer.pad_token]
                real_length = len(tokens)
                result_l2 = {
                    'index': example_indices[i],
                    'text': self.tokenizer.decode(input_ids[i], skip_special_tokens=True),
                        #'tokens': tokens,
                    'true_label': true_label,
                    'predicted_class': predicted_class,
                    'predicted_class_confidence': confidences[i][predicted_class],
                    'target_class': class_index,
                    'target_class_confidence': confidences[i][class_index],
                    'method': f"{self.method}_L2",
                    'attribution': list(zip(tokens, saliency_token_l2.tolist()[:real_length])),
                    'attribution_token': list(zip(tokens, saliency_token_l2.tolist()[:real_length])), 
                    'attribution_position': list(zip(tokens, saliency_position_l2.tolist()[:real_length])),
                    'attribution_token_type': list(zip(tokens, saliency_token_type_l2.tolist()[:real_length])),
                }

                result_mean = {
                    'index': example_indices[i],
                    'text': self.tokenizer.decode(input_ids[i], skip_special_tokens=True),
                        #'tokens': tokens,
                    'true_label': true_label,
                    'predicted_class': predicted_class,
                    'predicted_class_confidence': confidences[i][predicted_class],
                    'target_class': class_index,
                    'target_class_confidence': confidences[i][class_index],
                    'method': f"{self.method}_mean",
                    "attribution": list(zip(tokens, saliency_token_mean.tolist()[:real_length])),
                    'attribution_token': list(zip(tokens, saliency_token_mean.tolist()[:real_length])),
                    'attribution_position': list(zip(tokens, saliency_position_mean.tolist()[:real_length])),
                    'attribution_token_type': list(zip(tokens, saliency_token_type_mean.tolist()[:real_length])),
                }
                all_saliency_l2_results[i].append(result_l2)
                all_saliency_mean_results[i].append(result_mean)
        saliency_results = {f"{self.method}_L2": all_saliency_l2_results, f"{self.method}_mean": all_saliency_mean_results}
        return saliency_results
    
    def explain(self, texts, example_indices, labels=None, num_classes=None, class_labels=None, max_length=512):
        # if class_labels is not provided, then num_classes must be provided
        if class_labels is None:
            assert num_classes is not None, "Num_classes must be provided for explainer if class_labels is not provided"
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=self.device).unsqueeze(0).repeat(input_ids.size(0), 1)
        # if inputs has no 'token_type_ids' key, then token_type_ids = 0
        if 'token_type_ids' in inputs:
            token_type_ids = inputs['token_type_ids']
        else:
            token_type_ids = torch.zeros_like(input_ids)
        saliency_results = self._explain(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, token_type_ids=token_type_ids, example_indices=example_indices, labels=labels, num_classes=num_classes, class_labels=class_labels)
        return saliency_results
    
    def explain_hybrid_documents(self, text1, text2, example_indices, labels=None, num_classes=None, class_labels=None, max_length=512):
        # if class_labels is not provided, then num_classes must be provided
        if class_labels is None:
            assert num_classes is not None, "Num_classes must be provided for explainer if class_labels is not provided"
        inputs = self.tokenizer(text1, text2, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=self.device).unsqueeze(0).repeat(input_ids.size(0), 1)
        # token_type_ids: 0 for text1 and 1 for text2
        if 'token_type_ids' in inputs:
            token_type_ids = inputs['token_type_ids']
        else:
            token_type_ids = torch.zeros_like(input_ids)
            token_type_ids[:, input_ids.tolist().index(self.tokenizer.sep_token_id)] = 1
        saliency_results = self._explain(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, token_type_ids=token_type_ids, example_indices=example_indices, labels=labels, num_classes=num_classes, class_labels=class_labels)
        return saliency_results
    
    def explain_dataset(self, dataset, num_classes=None, class_labels=None, batch_size=16, max_length=512):
        # if class_labels is not provided, then num_classes must be provided
        if class_labels is None:
            assert num_classes is not None, "Num_classes must be provided for explainer if class_labels is not provided"
        data_loader = batch_loader(dataset, batch_size=batch_size, shuffle=False)
        saliency_l2_results = []
        saliency_mean_results = []
        class_labels_indexer = 0
        for batch in tqdm(data_loader):
            texts = batch['text']
            example_indices = batch['index']
            labels = batch['label']
            if class_labels is not None:
                batch_class_labels = [predicted_label[class_labels_indexer: class_labels_indexer+len(example_indices)] for predicted_label in class_labels]
                class_labels_indexer += len(example_indices)
            else:
                batch_class_labels = None
            saliency_results = self.explain(texts=texts, example_indices=example_indices, labels=labels, num_classes=num_classes, class_labels=batch_class_labels, max_length=max_length)
            saliency_l2_results.extend(saliency_results[f"{self.method}_L2"])
            saliency_mean_results.extend(saliency_results[f"{self.method}_mean"])
        saliency_results = {f"{self.method}_L2": saliency_l2_results, f"{self.method}_mean": saliency_mean_results}
        return saliency_results
    
    def explain_hybrid_documents_dataset(self, dataset, num_classes=None, class_labels=None, batch_size=16, max_length=512):
        # if class_labels is not provided, then num_classes must be provided
        if class_labels is None:
            assert num_classes is not None, "Num_classes must be provided for explainer if class_labels is not provided"
        data_loader = batch_loader(dataset, batch_size=batch_size, shuffle=False)
        saliency_l2_results = []
        saliency_mean_results = []
        class_labels_indexer = 0
        for batch in tqdm(data_loader):
            texts1 = batch['text1']
            texts2 = batch['text2']
            example_indices = batch['index']
            labels = None
            if class_labels is not None:
                batch_class_labels = [predicted_label[class_labels_indexer: class_labels_indexer+len(example_indices)] for predicted_label in class_labels]
                class_labels_indexer += len(example_indices)
            else:
                batch_class_labels = None
            saliency_results = self.explain_hybrid_documents(text1=texts1, text2=texts2, example_indices=example_indices, labels=labels, num_classes=num_classes, class_labels=batch_class_labels, max_length=max_length)
            saliency_l2_results.extend(saliency_results[f"{self.method}_L2"])
            saliency_mean_results.extend(saliency_results[f"{self.method}_mean"])
        saliency_results = {f"{self.method}_L2": saliency_l2_results, f"{self.method}_mean": saliency_mean_results}
        return saliency_results

class OcclusionExplainer(BaseExplainer):
    def __init__(self, model, tokenizer):
        self.model = BertModelWrapper(model)
        self.model.eval()
        self.model.to(model.device)
        self.tokenizer = tokenizer
        self.explainer = Occlusion(self.model)
        # Occlusion parameters
        self.sliding_window_size = (1,)  # Occlude one token at a time
        self.baseline = tokenizer.pad_token_id  # Use [PAD] token as baseline
        self.stride = (1,)
        self.device = model.device

    def _explain(self, input_ids, attention_mask, example_indices, labels=None, num_classes=None, class_labels=None):

        batch_size = input_ids.shape[0]
        # Get the model's predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_classes = outputs.argmax(dim=-1).detach().cpu().numpy().tolist()
        # confidence for each class
        confidences = torch.nn.functional.softmax(outputs, dim=-1).detach().cpu().numpy().tolist()

        all_explained_labels = []
        if class_labels is None:
            # explain for all classes
            for class_idx in range(num_classes):
                class_labels = [class_idx] * batch_size
                all_explained_labels.append(class_labels)
        else:
            all_explained_labels=class_labels

        all_occlusion_results = [[] for _ in range(batch_size)]
        for explained_labels in all_explained_labels:
            attributions = self.explainer.attribute(
                inputs=input_ids,
                strides=self.stride,
                sliding_window_shapes=self.sliding_window_size,
                baselines=self.baseline,
                target=explained_labels,
                additional_forward_args=(attention_mask,)
            )
            for i in range(batch_size):
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i].detach().cpu().numpy().tolist())
                class_index = explained_labels[i]
                predicted_class = predicted_classes[i]
                if labels is not None:
                    true_label = labels[i]
                else:
                    true_label = None
                attributions_i = attributions.detach().cpu().numpy()[i]  # Shape: [seq_len]
                # skip padding tokens
                tokens = [token for token in tokens if token != self.tokenizer.pad_token]
                real_length = len(tokens)
                # Collect results for the current example and class
                result = {
                    'index': example_indices[i],
                    'text': self.tokenizer.decode(input_ids[i], skip_special_tokens=True),
                    #'tokens': tokens,
                    'true_label': true_label,
                    'predicted_class': predicted_class,
                    'predicted_class_confidence': confidences[i][predicted_class],
                    'target_class': class_index,
                    'target_class_confidence': confidences[i][class_index],
                    'method': 'Occlusion',
                    'attribution': list(zip(tokens, attributions_i.tolist()[:real_length])),
                }
                all_occlusion_results[i].append(result)
        return {"Occlusion": all_occlusion_results}
    
    def explain(self, texts, example_indices, labels=None, num_classes=None, class_labels=None, max_length=512):
        # if class_labels is not provided, then num_classes must be provided
        if class_labels is None:
            assert num_classes is not None, "Num_classes must be provided for explainer if class_labels is not provided"
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        occlusion_results = self._explain(input_ids=input_ids, attention_mask=attention_mask, example_indices=example_indices, labels=labels, num_classes=num_classes, class_labels=class_labels)
        return occlusion_results
    
    def explain_hybrid_documents(self, text1, text2, example_indices, labels=None, num_classes=None, class_labels=None, max_length=512):
        # if class_labels is not provided, then num_classes must be provided
        if class_labels is None:
            assert num_classes is not None, "Num_classes must be provided for explainer if class_labels is not provided"
        inputs = self.tokenizer(text1, text2, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        occlusion_results = self._explain(input_ids=input_ids, attention_mask=attention_mask, example_indices=example_indices, labels=labels, num_classes=num_classes, class_labels=class_labels)
        return occlusion_results
    
    def explain_dataset(self, dataset, num_classes=None, class_labels=None, batch_size=16, max_length=512):
        # if class_labels is not provided, then num_classes must be provided
        if class_labels is None:
            assert num_classes is not None, "Num_classes must be provided for explainer if class_labels is not provided"
        data_loader = batch_loader(dataset, batch_size=batch_size, shuffle=False)
        occlusion_results = []
        class_labels_indexer = 0
        for batch in tqdm(data_loader):
            texts = batch['text']
            example_indices = batch['index']
            labels = batch['label']
            if class_labels is not None:
                batch_class_labels = [predicted_label[class_labels_indexer: class_labels_indexer+len(example_indices)] for predicted_label in class_labels]
                class_labels_indexer += len(example_indices)
            else:
                batch_class_labels = None
            occlusion_result = self.explain(texts=texts, example_indices=example_indices, labels=labels, num_classes=num_classes, class_labels=batch_class_labels, max_length=max_length)
            occlusion_results.extend(occlusion_result['Occlusion'])
        return {"Occlusion": occlusion_results}
    
    def explain_hybrid_documents_dataset(self, dataset, num_classes=None, class_labels=None, batch_size=16, max_length=512):
        # if class_labels is not provided, then num_classes must be provided
        if class_labels is None:
            assert num_classes is not None, "Num_classes must be provided for explainer if class_labels is not provided"
        data_loader = batch_loader(dataset, batch_size=batch_size, shuffle=False)
        occlusion_results = []
        class_labels_indexer = 0
        for batch in tqdm(data_loader):
            texts1 = batch['text1']
            texts2 = batch['text2']
            example_indices = batch['index']
            labels = None
            if class_labels is not None:
                batch_class_labels = [predicted_label[class_labels_indexer: class_labels_indexer+len(example_indices)] for predicted_label in class_labels]
                class_labels_indexer += len(example_indices)
            else:
                batch_class_labels = None
            occlusion_result = self.explain_hybrid_documents(text1=texts1, text2=texts2, example_indices=example_indices, labels=labels, num_classes=num_classes, class_labels=batch_class_labels, max_length=max_length)
            occlusion_results.extend(occlusion_result['Occlusion'])
        return {"Occlusion": occlusion_results}
    
class ShapleyValueExplainer(BaseExplainer):
    def __init__(self, model, tokenizer, n_samples=25):
        self.model = BertModelWrapper(model)
        self.model.eval()
        self.model.to(model.device)
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.explainer = ShapleyValueSampling(self.model)
        self.device = model.device

    def _explain(self, input_ids, attention_mask, example_indices, labels=None, num_classes=None, class_labels=None):

        batch_size = input_ids.shape[0]
        # Get the model's predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_classes = outputs.argmax(dim=-1).detach().cpu().numpy().tolist()
        # confidence for each class
        confidences = torch.nn.functional.softmax(outputs, dim=-1).detach().cpu().numpy().tolist()

        all_explained_labels = []
        if class_labels is None:
            # explain for all classes
            for class_idx in range(num_classes):
                class_labels = [class_idx] * batch_size
                all_explained_labels.append(class_labels)
        else:
            all_explained_labels=class_labels

        all_shapley_results = [[] for _ in range(batch_size)]
        for explained_labels in all_explained_labels:
            attributions = self.explainer.attribute(
                inputs=input_ids,
                baselines=None,
                target=explained_labels,
                additional_forward_args=(attention_mask,),
                feature_mask=None,
                n_samples=self.n_samples
            )
            for i in range(batch_size):
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i].detach().cpu().numpy().tolist())
                class_index = explained_labels[i]
                predicted_class = predicted_classes[i]
                if labels is not None:
                    true_label = labels[i]
                else:
                    true_label = None
                attributions_i = attributions.detach().cpu().numpy()[i]  # Shape: [seq_len]
                # skip padding tokens
                tokens = [token for token in tokens if token != self.tokenizer.pad_token]
                real_length = len(tokens)
                # Collect results for the current example and class
                result = {
                    'index': example_indices[i],
                    'text': self.tokenizer.decode(input_ids[i], skip_special_tokens=True),
                    #'tokens': tokens,
                    'true_label': true_label,
                    'predicted_class': predicted_class,
                    'predicted_class_confidence': confidences[i][predicted_class],
                    'target_class': class_index,
                    'target_class_confidence': confidences[i][class_index],
                    'method': 'ShapleyValue',
                    'attribution': list(zip(tokens, attributions_i.tolist()[:real_length])),
                }
                all_shapley_results[i].append(result)
        return {"ShapleyValue": all_shapley_results}
    
    def explain(self, texts, example_indices, labels=None, num_classes=None, class_labels=None, max_length=512):
        # if class_labels is not provided, then num_classes must be provided
        if class_labels is None:
            assert num_classes is not None, "Num_classes must be provided for explainer if class_labels is not provided"
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        shapley_results = self._explain(input_ids=input_ids, attention_mask=attention_mask, example_indices=example_indices, labels=labels, num_classes=num_classes, class_labels=class_labels)
        return shapley_results
    
    def explain_hybrid_documents(self, text1, text2, example_indices, labels=None, num_classes=None, class_labels=None, max_length=512):
        # if class_labels is not provided, then num_classes must be provided
        if class_labels is None:
            assert num_classes is not None, "Num_classes must be provided for explainer if class_labels is not provided"
        inputs = self.tokenizer(text1, text2, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        shapley_results = self._explain(input_ids=input_ids, attention_mask=attention_mask, example_indices=example_indices, labels=labels, num_classes=num_classes, class_labels=class_labels)
        return shapley_results
    
    def explain_dataset(self, dataset, num_classes=None, class_labels=None, batch_size=16, max_length=512):
        # if class_labels is not provided, then num_classes must be provided
        if class_labels is None:
            assert num_classes is not None, "Num_classes must be provided for explainer if class_labels is not provided"
        data_loader = batch_loader(dataset, batch_size=1, shuffle=False) # batch_size = 1 to avoid padding
        shapley_results = []
        class_labels_indexer = 0
        for batch in tqdm(data_loader):
            texts = batch['text']
            example_indices = batch['index']
            labels = batch['label']
            if class_labels is not None:
                batch_class_labels = [predicted_label[class_labels_indexer: class_labels_indexer+len(example_indices)] for predicted_label in class_labels]
                class_labels_indexer += len(example_indices)
            else:
                batch_class_labels = None
            shapley_result = self.explain(texts=texts, example_indices=example_indices, labels=labels, num_classes=num_classes, class_labels=batch_class_labels, max_length=max_length)
            shapley_results.extend(shapley_result['ShapleyValue'])
        return {"ShapleyValue": shapley_results}
    
    def explain_hybrid_documents_dataset(self, dataset, num_classes=None, class_labels=None, batch_size=16, max_length=512):
        # if class_labels is not provided, then num_classes must be provided
        if class_labels is None:
            assert num_classes is not None, "Num_classes must be provided for explainer if class_labels is not provided"
        data_loader = batch_loader(dataset, batch_size=1, shuffle=False)
        shapley_results = []
        class_labels_indexer = 0
        for batch in tqdm(data_loader):
            texts1 = batch['text1']
            texts2 = batch['text2']
            example_indices = batch['index']
            labels = None
            if class_labels is not None:
                batch_class_labels = [predicted_label[class_labels_indexer: class_labels_indexer+len(example_indices)] for predicted_label in class_labels]
                class_labels_indexer += len(example_indices)
            else:
                batch_class_labels = None
            shapley_result = self.explain_hybrid_documents(text1=texts1, text2=texts2, example_indices=example_indices, labels=labels, num_classes=num_classes, class_labels=batch_class_labels, max_length=max_length)
            shapley_results.extend(shapley_result['ShapleyValue'])
        return {"ShapleyValue": shapley_results}


class LimeExplainer(BaseExplainer):
    def __init__(self, model, tokenizer, batch_size=64, random_seed=42):
        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer
        self.random_seed = random_seed
        self.prob_func = BertProbabilityModelWrapper(self.model, self.tokenizer, batch_size=batch_size)
        self.device = model.device
               
    def explain(self, text, example_index, label=None, num_classes=None, class_label=None, max_length=512):
        # single instance
        inputs = self.tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        # Get the model's prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_class = outputs.logits.argmax(dim=-1).item()
            # confidence for each class
            confidences = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy().tolist()
        if class_label is None:
            # explain for all classes
            all_explained_labels = list(range(num_classes))
        else:
            all_explained_labels = [class_label]
        lime_results = []
        def bert_tokenizer(text):
            return self.tokenizer.tokenize(text)[:max_length-2] # subtract 2 for [CLS] and [SEP]
        tokenize_func = bert_tokenizer
        for explained_label in all_explained_labels:
            explanation = explain(text[0],
                predict_fn=self.prob_func,
                class_to_explain=explained_label,
                tokenizer=tokenize_func,
                seed=self.random_seed,
            )
        result = {
            'example_id': example_index,
            'text': ' '.join(explanation.features),
            #'tokens': explanation.features,
            'true_label': label,
            'predicted_class': predicted_class,
            'predicted_class_confidence': confidences[0][predicted_class],
            'target_class': explained_label,
            'target_class_confidence': confidences[0][explained_label],
            'method': 'Lime',
            'attribution': list(zip(explanation.features, explanation.feature_importance.tolist())),
        }
        lime_results.append(result)
        return {"Lime": [lime_results]}
    
    def explain_hybrid_documents(self, text1, text2, example_index, label=None, num_classes=None, class_label=None, max_length=512):
        # single instance
        inputs = self.tokenizer(text1, text2, truncation=True, max_length=max_length, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        # Get the model's prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_class = outputs.logits.argmax(dim=-1).item()
            # confidence for each class
            confidences = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy().tolist()
        if class_label is None:
            # explain for all classes
            all_explained_labels = list(range(num_classes))
        else:
            all_explained_labels = class_label
        lime_results = []
        def bert_tokenizer(text):
            return self.tokenizer.tokenize(text)[:max_length-2] # subtract 2 for [CLS] and [SEP]
        tokenize_func = bert_tokenizer
        # if num_classes is 2, run the explanation only once and generate explanation for the other class with minus attribution scores

        for explained_label in all_explained_labels:
            explanation = explain(' '.join([text1[0], text2[0]]),
                predict_fn=self.prob_func,
                class_to_explain=explained_label,
                tokenizer=tokenize_func,
                seed=self.random_seed,
            )
            result = {
                'example_id': example_index[0],
                'text': ' '.join(explanation.features),
                #'tokens': explanation.features,
                'true_label': None,
                'predicted_class': predicted_class,
                'predicted_class_confidence': confidences[0][predicted_class],
                'target_class': explained_label,
                'target_class_confidence': confidences[0][explained_label],
                'method': 'Lime',
                'attribution': list(zip(explanation.features, explanation.feature_importance.tolist())),
            }
            lime_results.append(result)
            if num_classes == 2:
                neg_result = {
                    'example_id': example_index[0],
                    'text': ' '.join(explanation.features),
                    #'tokens': explanation.features,
                    'true_label': label[0],
                    'predicted_class': predicted_class,
                    'predicted_class_confidence': confidences[0][predicted_class],
                    'target_class': all_explained_labels[1],
                    'target_class_confidence': 1 - confidences[0][explained_label],
                    'method': 'Lime',
                    'attribution': list(zip(explanation.features, [-x for x in explanation.feature_importance.tolist()])),
                }
                lime_results.append(neg_result)
                break
        return {"Lime": [lime_results]}
        

    def explain_dataset(self, dataset, num_classes=None, class_labels=None, batch_size=16, max_length=512):
        lime_results = []
        class_labels_indexer = 0
        for example in tqdm(batch_loader(dataset, batch_size=1, shuffle=False)): # batch operation is not supported now
            example_index = example['index']
            text = example['text']
            label = example['label']
            batch_class_labels = [predicted_label[class_labels_indexer: class_labels_indexer+len(example_index)] for predicted_label in class_labels]
            batch_class_labels = [predicted_label[0] for predicted_label in batch_class_labels]
            class_labels_indexer += len(example_index)
            lime_result = self.explain(text=text, example_index=example_index, label=label, num_classes=num_classes, class_label=batch_class_labels, max_length=max_length)
            lime_results.extend(lime_result['Lime'])
        return {"Lime": lime_results}
    
    def explain_hybrid_documents_dataset(self, dataset, num_classes=None, class_labels=None, batch_size=16, max_length=512):
        lime_results = []
        class_labels_indexer = 0
        for example in tqdm(batch_loader(dataset, batch_size=1, shuffle=False)):
            example_index = example['index']
            text1 = example['text1']
            text2 = example['text2']
            label = None
            batch_class_labels = [predicted_label[class_labels_indexer: class_labels_indexer+len(example_index)] for predicted_label in class_labels]
            batch_class_labels = [predicted_label[0] for predicted_label in batch_class_labels]
            class_labels_indexer += len(example_index)
            lime_result = self.explain_hybrid_documents(text1=text1, text2=text2, example_index=example_index, label=label, num_classes=num_classes, class_label=batch_class_labels, max_length=max_length)
            lime_results.extend(lime_result['Lime'])
        return {"Lime": lime_results}

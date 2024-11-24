import argparse
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from captum.attr import Saliency, DeepLift, GuidedBackprop, InputXGradient, IntegratedGradients, Occlusion, ShapleyValueSampling, DeepLiftShap, GradientShap, KernelShap 
from saliency_utils.lime_utils import explain
from datasets import load_dataset
import numpy as np
import json
import os
import random
from tqdm import tqdm
from saliency_utils.utils import batch_loader


class BcosBertEmbeddingModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(BcosBertEmbeddingModelWrapper, self).__init__()
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
        pooled_output = self.model.dropout(pooled_output)
        logits = self.model.classifier(pooled_output)
        return logits

class RelativeBcosBertEmbeddingModelWrapper(BcosBertEmbeddingModelWrapper):
    def __init__(self, model):
        super(RelativeBcosBertEmbeddingModelWrapper, self).__init__(model)

    def forward(self, token_embeddings, position_embeddings, token_type_embeddings, attention_mask=None):
        logits = super().forward(token_embeddings, position_embeddings, token_type_embeddings, attention_mask)
        return logits - logits.mean(dim=1, keepdim=True)


    
class BaseExplainer:
    def explain(self):
        raise NotImplementedError


class BcosExplainer(BaseExplainer):
    def __init__(self, model, tokenizer, relative=True):
        if relative:
            self.model = RelativeBcosBertEmbeddingModelWrapper(model)
        else:
            self.model = BcosBertEmbeddingModelWrapper(model)
        self.model.eval()
        self.model.to(model.device)
        self.tokenizer = tokenizer
        self.device = model.device
        #self.explainer = InputXGradient(self.model)
        self.method = "Bcos_relative" if relative else "Bcos_absolute"
    
    def _explain(self, input_ids, attention_mask, position_ids=None, token_type_ids=None, example_indices=None, labels=None, num_classes=None, class_labels=None, only_predicted_classes=False):

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
        if class_labels is None and num_classes is not None:           
            # explain for all classes
            for class_idx in range(num_classes):
                class_labels = [class_idx] * batch_size
                all_explained_labels.append(class_labels)
        else:
            all_explained_labels=class_labels
        
        if only_predicted_classes:
            all_explained_labels = [predicted_classes]

        all_saliency_l2_results = [[] for _ in range(batch_size)]
        all_saliency_mean_results = [[] for _ in range(batch_size)]
        for explained_labels in all_explained_labels:
            # activate explanation mode
            with self.model.model.explanation_mode():
                explainer = InputXGradient(self.model)
                attributions = explainer.attribute(
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
    
    def explain(self, texts, example_indices, labels=None, num_classes=None, class_labels=None, max_length=512, only_predicted_classes=False):
        # if class_labels is not provided, then num_classes must be provided
        if class_labels is None:
            assert num_classes is not None or only_predicted_classes, "Num_classes must be provided for explainer if class_labels is not provided"
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
        saliency_results = self._explain(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, token_type_ids=token_type_ids, example_indices=example_indices, labels=labels, num_classes=num_classes, class_labels=class_labels, only_predicted_classes=only_predicted_classes)
        return saliency_results
    
    def explain_hybrid_documents(self, text1, text2, example_indices, labels=None, num_classes=None, class_labels=None, max_length=512, only_predicted_classes=False):
        # if class_labels is not provided, then num_classes must be provided
        if class_labels is None:
            assert num_classes is not None or only_predicted_classes, "Num_classes must be provided for explainer if class_labels is not provided"
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
        saliency_results = self._explain(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, token_type_ids=token_type_ids, example_indices=example_indices, labels=labels, num_classes=num_classes, class_labels=class_labels, only_predicted_classes=only_predicted_classes)
        return saliency_results
    
    def explain_dataset(self, dataset, num_classes=None, class_labels=None, batch_size=16, max_length=512, only_predicted_classes=False):
        # if class_labels is not provided, then num_classes must be provided
        if class_labels is None:
            assert num_classes is not None or only_predicted_classes, "Num_classes must be provided for explainer if class_labels is not provided"
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
            saliency_results = self.explain(texts=texts, example_indices=example_indices, labels=labels, num_classes=num_classes, class_labels=batch_class_labels, max_length=max_length, only_predicted_classes=only_predicted_classes)
            saliency_l2_results.extend(saliency_results[f"{self.method}_L2"])
            saliency_mean_results.extend(saliency_results[f"{self.method}_mean"])
        saliency_results = {f"{self.method}_L2": saliency_l2_results, f"{self.method}_mean": saliency_mean_results}
        return saliency_results
    
    def explain_hybrid_documents_dataset(self, dataset, num_classes=None, class_labels=None, batch_size=16, max_length=512, only_predicted_classes=False):
        # if class_labels is not provided, then num_classes must be provided
        if class_labels is None:
            assert num_classes is not None or only_predicted_classes, "Num_classes must be provided for explainer if class_labels is not provided"
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
            saliency_results = self.explain_hybrid_documents(text1=texts1, text2=texts2, example_indices=example_indices, labels=labels, num_classes=num_classes, class_labels=batch_class_labels, max_length=max_length, only_predicted_classes=only_predicted_classes)
            saliency_l2_results.extend(saliency_results[f"{self.method}_L2"])
            saliency_mean_results.extend(saliency_results[f"{self.method}_mean"])
        saliency_results = {f"{self.method}_L2": saliency_l2_results, f"{self.method}_mean": saliency_mean_results}
        return saliency_results



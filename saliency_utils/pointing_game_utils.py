import torch
from torch.utils.data import Subset
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizer, AutoConfig
from bcos_lm.models.modeling_bert import BertForSequenceClassification
import json
import random
import os
from tqdm import tqdm
from saliency_utils.utils import set_random_seed, split_dataset, batch_loader
from saliency_utils.Explainer import BcosExplainer, AttentionExplainer, GradientNPropabationExplainer, OcclusionExplainer, ShapleyValueExplainer, LimeExplainer

EXPLANATION_METHODS = {
    "Bcos": BcosExplainer,
    "Attention": AttentionExplainer,
    "Saliency": GradientNPropabationExplainer,
    "DeepLift": GradientNPropabationExplainer,
    "GuidedBackprop": GradientNPropabationExplainer,
    "InputXGradient": GradientNPropabationExplainer,
    "IntegratedGradients": GradientNPropabationExplainer,
    "Occlusion": OcclusionExplainer,
    "ShapleyValue": ShapleyValueExplainer,
    "GradientShap": ShapleyValueExplainer,
    "DeepLiftShap": ShapleyValueExplainer,
    "KernelShap": ShapleyValueExplainer,
    "Lime": LimeExplainer,
}

class GridPointingGame:
    def __init__(
            self,
            model_name_or_path,
            dataset,
            num_labels,
            split='test',
            split_ratio=0.5,
            #embedding_attributions=None,
            load_pointing_game_examples_path=None,
            save_pointing_game_examples_path=None,
            num_segments=2,
            max_length=128,
            batch_size=16,
            num_instances=-1,
            min_confidence=0.5,
            random_seed=42,
            #bcos=False,
            #b=2.0,
    ):
        """
        Filter and truncate dataset
        Compute the confidence of model predictions
        Sample and create pointing game instances 
        """
        assert num_segments == 2, "Currently only support 2 segments"
        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        config.num_labels = num_labels
        #config.bcos = bcos,
        #config.b = b
        config.output_attentions = True
        self.model = BertForSequenceClassification.load_from_pretrained(model_name_or_path, config=config)
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        set_random_seed(random_seed)
        self.num_labels = num_labels
        self.num_segments = num_segments
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_instances = num_instances
        self.min_confidence = min_confidence
        #self.embedding_attributions = embedding_attributions
        #self.bcos = bcos
        #self.b = b


        # detect string None
        if load_pointing_game_examples_path is None or load_pointing_game_examples_path == "None" or not os.path.exists(load_pointing_game_examples_path):
            load_pointing_game_examples_path = None
        if save_pointing_game_examples_path == "None":
            save_pointing_game_examples_path = None

        # load or create pointing game instances
        if load_pointing_game_examples_path is not None:
            self.instances = self.load_from_file(load_pointing_game_examples_path)

            # take num_instances for testing
            if self.num_instances > 0:
                self.instances = {key: self.instances[key][:self.num_instances] for key in self.instances.keys()}

            # get the number of instances
            self.num_instances = len(self.instances[list(self.instances.keys())[0]])
        else: 
            # load and split dataset               
            dataset = load_dataset(dataset, split=split)
            if split_ratio is not None:
                dataset = split_dataset(dataset, split_ratio)
            # filter and truncate dataset
            truncated_dataset = self.truncate_dataset(dataset)
            # compute and sort the confidence of model predictions
            sorted_confidence_results = self.sort_by_confidence(truncated_dataset)
            # sample and create pointing game instances

            self.instances = self.sample_pointing_game_instances(sorted_confidence_results, num_instances)
            #self.instances = self.sample_pointing_game_instances_w_fixed_order(sorted_confidence_results, num_instances)
            # save pointing game instances
            if save_pointing_game_examples_path is not None:
                self.save_to_file(self.instances, save_pointing_game_examples_path)

    @staticmethod           
    def save_to_file(data, path):
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved data to {path}")

    @staticmethod
    def load_from_file(path):
        with open(path, "r") as f:
            data = json.load(f)
        print(f"Loaded data from {path}")
        return data

    def truncate_dataset(self, dataset):
        """
        Keep only examples longer than the max length and truncate them.
        Args:
            dataset: The dataset to truncate.
        Returns:
            truncated_dataset (dict): The truncated dataset.
        """
        texts = []
        labels = []

        for example in tqdm(dataset):
            text, label = example["text"], example["label"]
        #for text, label in zip(dataset["text"], dataset["label"]):
            tokens = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length")
            if tokens["input_ids"].count(self.tokenizer.pad_token_id) == 0: # Filter out examples with less than max_length tokens
                truncated_text = self.tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)
                texts.append(truncated_text)
                labels.append(label)

        print(f"Truncated dataset size: {len(texts)}")
        return {"text": texts, "label": labels}

    def _compute_prediction_confidence(self, input_ids, attention_mask):
        """
        Compute the confidence of the model predictions.

        Args:
            model: The BERT model.
            input_ids (torch.Tensor): Input token IDs. Shape: [batch_size, seq_length]
            attention_mask (torch.Tensor): Attention mask. Shape: [batch_size, seq_length]

        Returns:
            prediction (List): Predicted label indices. 
            confidence (List): Confidence scores for the prediction of each example. 
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=1).detach().cpu().numpy().tolist()
            confidences = torch.max(probs, dim=1).values.detach().cpu().numpy().tolist()

        return predictions, confidences

    def compute_prediction_confidence(self, dataset):
        """
        Compute the confidence of the model predictions on a dataset.

        Args:
            model: The BERT model.
            tokenizer: The tokenizer used to tokenize the dataset.
            dataset (torch.utils.data.Dataset): The original text dataset to compute the prediction confidence.
            max_length (int): Maximum length to truncate the dataset.
            batch_size (int): Batch size for computing the confidence.

        Returns:
            orig_texts (Dict): Original truncated text examples.
            labels (List): True label indices.
            predictions (List): Predicted label indices. 
            confidences (List): Confidence scores for the prediction of each example. 
        """
        orig_texts = dataset["text"]
        labels = dataset["label"]
        predictions = []
        confidences = []
        dataloader = batch_loader(dataset, batch_size=self.batch_size, shuffle=False)
        for batch in tqdm(dataloader):
            inputs = self.tokenizer(batch["text"], truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            batch_predictions, batch_confidences = self._compute_prediction_confidence(input_ids, attention_mask)
            predictions.extend(batch_predictions)
            confidences.extend(batch_confidences)
        print("Finished computing prediction confidence.")
        return orig_texts, labels, predictions, confidences

    def sort_by_confidence(self, dataset):
        """
        Sort the examples by confidence.
        Collect the top examples for each label that are correctly predicted with the highest confidence. The confidence threshold is set to 0.5.
        Args:
            orig_texts (List): Original truncated text examples.
            labels (List): True label indices.
            predictions (List): Predicted label indices. 
            confidences (List): Confidence scores for the prediction of each example. 
            num_labels (int): Number of labels in the dataset.
            top_examples (int): Number of top examples to return.

        Returns:
            sorted_output (Dict): Dictionary containing the top examples sorted by confidence. Keys are class labels and values include:
            top_texts (List): Top text examples sorted by confidence.
            top_predictions (List): Predicted label indices for the top examples. 
            top_confidences (List): Confidence scores for the prediction of the top examples. 
        """
        orig_texts, labels, predictions, confidences = self.compute_prediction_confidence(dataset)
        sorted_output = {i: {"texts": [], "predictions": [], "confidences": []} for i in range(self.num_labels)}
        for label in range(self.num_labels):
            # select the top examples that are correctly predicted with the highest confidence for the current label
            label_indices = [i for i, pred in enumerate(predictions) if pred == label and labels[i] == label]
            label_confidences = [confidences[i] for i in label_indices]
            sorted_indices = np.argsort(label_confidences)[::-1]
            # filter out examples with confidence smaller than min_confidence
            sorted_indices = [i for i in sorted_indices if label_confidences[i] > self.min_confidence]
            sorted_output[label]["texts"] = [orig_texts[label_indices[i]] for i in sorted_indices]
            sorted_output[label]["predictions"] = [predictions[label_indices[i]] for i in sorted_indices]
            sorted_output[label]["confidences"] = [confidences[label_indices[i]] for i in sorted_indices]
        print("Finished sorting examples by confidence.")
        return sorted_output

    def sample_pointing_game_instances(self, sorted_output, num_instances=-1):
        # for each instance, randomly sample n_segments classes that still have unselected examples and select the next example from the class with the highest confidence
        # each instance contains n_segments segments of text, each corresponding to a different class
        # output: list of instances (tuple of segments), list of labels (tuple of labels)
        instances = []
        classes = []
        confidences = []
        class_indexer = {i: 0 for i in range(self.num_labels)}
        # if num_instances=-1, sample as many instances as possible until no valid instances can be created
        if num_instances == -1:
            while True:
                instance, label, confidence = self.sample_instance(sorted_output, class_indexer)
                if instance is None:
                    break
                instances.append(instance)
                classes.append(label)
                confidences.append(confidence)
        else:
            for _ in range(num_instances):
                instance, label, confidence = self.sample_instance(sorted_output, class_indexer)
                if instance is None:
                    break
                instances.append(instance)
                classes.append(label)
                confidences.append(confidence)
        dataset = {"index": list(range(len(instances))),"text1": [instance[0] for instance in instances], "text2": [instance[1] for instance in instances], "label1": [label[0] for label in classes], "label2": [label[1] for label in classes], "confidence1": [confidence[0] for confidence in confidences], "confidence2": [confidence[1] for confidence in confidences]}
        print(f"Number of pointing game instances: {len(instances)}")
        return dataset

    def sample_instance(self, sorted_output, class_indexer):
        # sample a single instance which consists of n_segments segments of text, each corresponding to a different class
        instance = []
        label = []
        confidence = []
        for _ in range(self.num_segments):
            valid_classes = [i for i in range(self.num_labels) if class_indexer[i] < len(sorted_output[i]["texts"]) and i not in label]
            if len(valid_classes) == 0:
                return None, None
            selected_class = random.choice(valid_classes)
            selected_index = class_indexer[selected_class]
            instance.append(sorted_output[selected_class]["texts"][selected_index])
            label.append(selected_class)
            confidence.append(sorted_output[selected_class]["confidences"][selected_index])
            class_indexer[selected_class] += 1
        return instance, label, confidence

    def initialize_explainer(self, method, model, tokenizer, baseline='pad', n_samples=25, relative=True):
        if EXPLANATION_METHODS[method] == BcosExplainer:
            explainer = BcosExplainer(model=model, tokenizer=tokenizer, relative=relative)
        elif EXPLANATION_METHODS[method] == ShapleyValueExplainer:
            explainer = ShapleyValueExplainer(model=model, tokenizer=tokenizer, method=method, baseline=baseline, n_samples=n_samples)
        # for GradientNPropabationExplainer, we need to specify the method
        elif EXPLANATION_METHODS[method] == GradientNPropabationExplainer:
            explainer = EXPLANATION_METHODS[method](model=model, tokenizer=tokenizer, method=method, baseline=baseline)
        else:
            explainer = EXPLANATION_METHODS[method](model=model, tokenizer=tokenizer) 
        return explainer
    
    def run_analysis(self, method_name, n_samples=None, load_explanations_path=None, save_explanations_path=None, save_evaluation_results_path=None, baseline='pad', relative=True):
        # encode the instances with multiple segments
        # compute the saliency scores for each segment for each class
        # look at how much positive saliency is assigned to the correct class; and whether the largest attribution is assigned to the correct class
        # output: list of positive saliency scores for each segment for each class, list of whether the largest attribution is assigned to the correct class
        if load_explanations_path is None or load_explanations_path == "None" or not os.path.exists(load_explanations_path):
            load_explanations_path = None
       
        if load_explanations_path is not None:
            explanations = self.load_from_file(load_explanations_path)
        else:
            explainer = self.initialize_explainer(method=method_name, model=self.model, tokenizer=self.tokenizer, baseline=baseline, n_samples=n_samples, relative=relative)
            explanations = self.explain_instances(explainer, save_explanations_path)
        evaluation_results = self.evaluate_explanations(explanations, save_evaluation_results_path)
        return evaluation_results

    def explain_instances(self, explainer, save_explanations_path=None):
        # encode the instances with multiple segments
        # compute the saliency scores for each segment for each class
        # save the explanations
        if save_explanations_path == "None":
            save_explanations_path = None
        class_labels = [self.instances['label1'], self.instances['label2']]
        explanations = explainer.explain_hybrid_documents_dataset(self.instances, num_classes=self.num_labels, batch_size=self.batch_size, class_labels=class_labels, max_length=2 * self.max_length-1)
        if save_explanations_path is not None:
            self.save_to_file(explanations, save_explanations_path)
        return explanations

    def evaluate_explanations(self, explanations, save_evaluation_results_path=None):
        # look at how much positive saliency is assigned to the correct class; and whether the largest attribution is assigned to the correct class
        # output: list of positive saliency scores for each segment for each class, list of whether the largest attribution is assigned to the correct class
        if save_evaluation_results_path == "None":
            save_evaluation_results_path = None
        all_evaluation_results = {}
        for method in explanations.keys():
            all_evaluation_results[method] = {}
            print(f"Method: {method}")
            method_explanations = explanations[method]
                
            # each instance should correspond to num_labels explanations
            correct_attribution = []
            correct_largest_attribution = []
            correct_positive_attribution = []
            correct_num_positive_attribution = []
            correct_total_attribution = []
            for idx in range(self.num_instances):
                label1 = self.instances['label1'][idx]
                label2 = self.instances['label2'][idx]
                
                # find the explanations which predicts label1 and label2
                explanation1 = method_explanations[idx][0]
                explanation2 = method_explanations[idx][1]

                assert explanation1['target_class'] == label1
                assert explanation2['target_class'] == label2

                
                correct_largest_attribution1, correct_positive_attribution1, correct_num_positive_attribution1, correct_attribution_ratio1, correct_total_attribution1 = self.evaluate_single_explanation(explanation1, 0)
                correct_largest_attribution2, correct_positive_attribution2, correct_num_positive_attribution2, correct_attribution_ratio2, correct_total_attribution2 = self.evaluate_single_explanation(explanation2, 1)
                correct_attribution.append([correct_attribution_ratio1, correct_attribution_ratio2])
                correct_largest_attribution.append([correct_largest_attribution1, correct_largest_attribution2])
                correct_positive_attribution.append([correct_positive_attribution1, correct_positive_attribution2])
                correct_num_positive_attribution.append([correct_num_positive_attribution1, correct_num_positive_attribution2])
                correct_total_attribution.append([correct_total_attribution1, correct_total_attribution2])

            # compute the average positive saliency scores for instance

            correct_attribution_per_instance = np.mean(correct_attribution, axis=1).tolist()
            correct_positive_attribution_per_instance = np.mean(correct_positive_attribution, axis=1).tolist()
            correct_largest_attribution_per_instance = np.mean(correct_largest_attribution, axis=1).tolist()
            correct_num_positive_attribution_per_instance = np.mean(correct_num_positive_attribution, axis=1).tolist()
            correct_total_attribution_per_instance = np.mean(correct_total_attribution, axis=1).tolist()

            # compute the average positive saliency scores for all instances
            correct_attribution_ratio = np.mean(correct_attribution_per_instance)
            correct_positive_attribution_ratio = np.mean(correct_positive_attribution_per_instance)
            correct_largest_attribution_ratio = np.mean(correct_largest_attribution_per_instance)
            correct_num_positive_attribution_ratio = np.mean(correct_num_positive_attribution_per_instance)
            correct_total_attribution_ratio = np.mean(correct_total_attribution_per_instance)

            print(f"Correct attribution ratio: {correct_attribution_ratio}")
            print(f"Correct positive saliency ratio: {correct_positive_attribution_ratio}")
            print(f"Correct largest attribution ratio: {correct_largest_attribution_ratio}")
            print(f"Correct number of positive saliency ratio: {correct_num_positive_attribution_ratio}")
            print(f"Correct total attribution ratio: {correct_total_attribution_ratio}")

            all_evaluation_results[method]["average_correct_attribution"] = correct_attribution_ratio
            all_evaluation_results[method]["average_correct_positive_attribution"] = correct_positive_attribution_ratio
            all_evaluation_results[method]["average_correct_largest_attribution"] = correct_largest_attribution_ratio
            all_evaluation_results[method]["average_correct_num_positive_attribution"] = correct_num_positive_attribution_ratio
            all_evaluation_results[method]["average_correct_total_attribution"] = correct_total_attribution_ratio

            all_evaluation_results[method]["correct_attribution_per_instance"] = correct_attribution_per_instance
            all_evaluation_results[method]["correct_positive_attribution_per_instance"] = correct_positive_attribution_per_instance
            all_evaluation_results[method]["correct_largest_attribution_per_instance"] = correct_largest_attribution_per_instance
            all_evaluation_results[method]["correct_num_positive_attribution_per_instance"] = correct_num_positive_attribution_per_instance
            all_evaluation_results[method]["correct_total_attribution_per_instance"] = correct_total_attribution_per_instance

            all_evaluation_results[method]["correct_attribution"] = correct_attribution
            all_evaluation_results[method]["correct_positive_attribution"] = correct_positive_attribution
            all_evaluation_results[method]["correct_largest_attribution"] = correct_largest_attribution
            all_evaluation_results[method]["correct_num_positive_attribution"] = correct_num_positive_attribution
            all_evaluation_results[method]["correct_total_attribution"] = correct_total_attribution
            
        if save_evaluation_results_path is not None:
            self.save_to_file(all_evaluation_results, save_evaluation_results_path)
        return all_evaluation_results
    
    def evaluate_single_explanation(self, explanation, correct_segment):
        # look at how much positive saliency is assigned to the correct class; and whether the largest attribution is assigned to the correct class
        # output: positive saliency scores for the correct segment, whether the largest attribution is assigned to the correct segment
        
        # when not specified or any element in the list is not a key in the explanation, use the overall attribution
        
        attribution = explanation['attribution']


        attribution_scores = [attr[1] for attr in attribution if attr[0]!=self.tokenizer.pad_token and attr[0]!=self.tokenizer.cls_token and attr[0]!=self.tokenizer.sep_token]
        if len(attribution_scores) != 2 * self.max_length - 4:
            print(f"Warning: the length of the attribution scores is not correct: {len(attribution_scores)}")
        sep_position = len(attribution_scores) // 2
        # find the largest attribution position
        largest_attribution_position = np.argmax(attribution_scores)
        # compare the total positive saliency in each segment, consider only the positive saliency
        positive_attribution_scores = [max(0, score) for score in attribution_scores]
        if correct_segment == 0:
            correct_largest_attribution = 1 if largest_attribution_position < sep_position else 0
            correct_positive_attribution_ratio = np.sum(positive_attribution_scores[:sep_position]) / (np.sum(positive_attribution_scores) + 1e-12)
            correct_attribution_ratio = np.sum(attribution_scores[:sep_position]) / (np.sum(attribution_scores) + 1e-12)
            # compare number of positive saliency in each segment
            if np.sum([1 for score in positive_attribution_scores if score > 0]) == 0:
                correct_num_positive_attribution_ratio = 0.5
            else:
                correct_num_positive_attribution_ratio = np.sum([1 for score in positive_attribution_scores[:sep_position] if score > 0]) / np.sum([1 for score in positive_attribution_scores if score > 0])
        elif correct_segment == 1:
            correct_largest_attribution = 1 if largest_attribution_position >= sep_position else 0
            correct_positive_attribution_ratio = np.sum(positive_attribution_scores[sep_position:]) / (np.sum(positive_attribution_scores) + 1e-12)
            correct_attribution_ratio = np.sum(attribution_scores[sep_position:]) / (np.sum(attribution_scores) + 1e-12)
            # compare number of positive saliency in each segment
            if np.sum([1 for score in positive_attribution_scores if score > 0]) == 0:
                correct_num_positive_attribution_ratio = 0.5
            else:
                correct_num_positive_attribution_ratio = np.sum([1 for score in positive_attribution_scores[sep_position:] if score > 0]) / np.sum([1 for score in positive_attribution_scores if score > 0])
        else: 
            raise ValueError("Currently only support 2 segments")
        correct_total_attribution = 1 if correct_attribution_ratio > 0.5 else 0
        return correct_largest_attribution, correct_positive_attribution_ratio, correct_num_positive_attribution_ratio, correct_attribution_ratio, correct_total_attribution
        







    def sample_pointing_game_instances_w_fixed_order(self, sorted_output, num_instances=-1):
        # for each instance, randomly sample n_segments classes that still have unselected examples and select the next example from the class with the highest confidence
        # each instance contains n_segments segments of text, each corresponding to a different class
        # output: list of instances (tuple of segments), list of labels (tuple of labels)
        instances = []
        classes = []
        confidences = []
        class_indexer = {i: 0 for i in range(self.num_labels)}
        # if num_instances=-1, sample as many instances as possible until no valid instances can be created
        if num_instances == -1:
            while True:
                instance, label, confidence = self.sample_instance_w_fixed_order(sorted_output, class_indexer)
                if instance is None:
                    break
                instances.append(instance)
                classes.append(label)
                confidences.append(confidence)
        else:
            for _ in range(num_instances):
                instance, label, confidence = self.sample_instance_w_fixed_order(sorted_output, class_indexer)
                if instance is None:
                    break
                instances.append(instance)
                classes.append(label)
                confidences.append(confidence)
        # reverse order of the instances
        
        dataset = {"index": list(range(2*len(instances))),"text1": [instance[0] for instance in instances]+[instance[1] for instance in instances], "text2": [instance[1] for instance in instances]+[instance[0] for instance in instances], "label1": [label[0] for label in classes]+[label[1] for label in classes], "label2": [label[1] for label in classes]+[label[0] for label in classes], "confidence1": [confidence[0] for confidence in confidences]+[confidence[1] for confidence in confidences], "confidence2": [confidence[1] for confidence in confidences]+[confidence[0] for confidence in confidences]}
        print(f"Number of pointing game instances: {len(instances)}")
        return dataset

    def sample_instance_w_fixed_order(self, sorted_output, class_indexer):
        # sample a single instance which consists of n_segments segments of text, each corresponding to a different class
        instance = []
        label = []
        confidence = []
        for _ in range(self.num_segments):
            valid_classes = [i for i in range(self.num_labels) if class_indexer[i] < len(sorted_output[i]["texts"]) and i not in label]
            if len(valid_classes) == 0:
                return None, None
            selected_class = 0 if len(label) == 0 else 1
            selected_index = class_indexer[selected_class]
            instance.append(sorted_output[selected_class]["texts"][selected_index])
            label.append(selected_class)
            confidence.append(sorted_output[selected_class]["confidences"][selected_index])
            class_indexer[selected_class] += 1
        return instance, label, confidence





                

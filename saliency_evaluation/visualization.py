import numpy as np
import matplotlib
from IPython.display import HTML, display
import json
from tqdm import tqdm

def print_importance(html, importances, tokenized_texts, idx, true_label, predicted_class, discrete=False, prefixes="", no_cls_sep=False):
    """
    importance: (sent_len)
    """
    if html is None or html == "":
        html = "<pre style='color:black; padding: 3px;'>\n"
    else:
        html += '\n' 
    html += f"Example {idx}\n"
    html += f"True label: {true_label}, Predicted class: {predicted_class}\n"
    for importance, tokenized_text, prefix in zip(importances, tokenized_texts, prefixes):
        if no_cls_sep:
            importance = importance[1:-1]
            tokenized_text = tokenized_text[1:-1]
        importance = importance / np.abs(importance).max() / 1.5  # Normalize
        if discrete:
            importance = np.argsort(np.argsort(importance)) / len(importance) / 1.6
        
        html += prefix
        html += "\n"

        for i in range(len(tokenized_text)):
            if importance[i] >= 0:
                rgba = matplotlib.colormaps.get_cmap('Greens')(importance[i])   # Wistia
            else:
                rgba = matplotlib.colormaps.get_cmap('Reds')(np.abs(importance[i]))   # Wistia
            text_color = "color: rgba(255, 255, 255, 1.0); " if np.abs(importance[i]) > 0.9 else ""
            color = f"background-color: rgba({rgba[0]*255}, {rgba[1]*255}, {rgba[2]*255}, {rgba[3]}); " + text_color
            html += (f"<span style='"
                    f"{color}"
                    f"border-radius: 5px; padding: 3px;"
                    f"font-weight: {int(800)};"
                    "'>")
            html += tokenized_text[i].replace('<', "[").replace(">", "]")
            html += "</span> "
        html += "\n"
    #display(HTML(html))
    return html

def print_importance_dataset(explanation_path, method, save_path, no_cls_sep=False, num_examples=-1):
    with open(explanation_path, 'r') as f:
        explanations = json.load(f)[method]
    
    if num_examples > 0:
        explanations = explanations[:num_examples]
    html = None
    for example in tqdm(explanations):
        true_label = example[0]["true_label"]
        predicted_class = example[0]["predicted_class"]
        idx = example[0]["index"]
        tokenized_texts = []
        attribution_scores = []
        prefixes = []
        for expl_id in range(len(example)):
            tokens = [attr[0] for attr in example[expl_id]["attribution"]]
            explained_label = example[expl_id]["target_class"]
            tokenized_texts.append(tokens)
            attribution_scores.append([attr[1] for attr in example[expl_id]["attribution"]])
            prefixes.append(f"Target Class {explained_label}:\t")           
        html = print_importance(html=html, importances=attribution_scores, tokenized_texts=tokenized_texts, idx=idx, true_label=true_label, predicted_class=predicted_class, discrete=False, prefixes=prefixes, no_cls_sep=no_cls_sep)

    with open(save_path, 'w') as f:
        f.write(html)
    
    return html

if __name__ == "__main__":
    import os
    dataset = "hatexplain"
    model = "bert"
   
    b = "2_5"
    explanation_path = f"/scratch/yifwang/rerun_results_42/same_set_pointing_game_{dataset}_{model}_bce_different_b_{b}_seed_42_train_all/Bcos_explanations.json"
    method = "Bcos_absolute_ixg_mean"
    no_cls_sep = False
    save_path = f"pg_{dataset}_{model}_explanation_b_{b}_visualization.html"
    explanation_path = "ablation/perturbation_hatexplain_bert_base_orig_train_all/Bcos_explanations.json"
    save_path = "visualization_hatexplain/visualization_baseline/hatexplain_bert_bcos_visualization.html"
    num_examples = -1

    html = print_importance_dataset(explanation_path, method, save_path, no_cls_sep=no_cls_sep, num_examples=num_examples)
    
    """
    if os.path.exists(f"visualization_baseline") is False:
        os.mkdir(f"visualization_baseline")
    methods = ['decompx', 'Attention', "DeepLift", "InputXGradient", "IntegratedGradients", "KernelShap", "Lime", "Occlusion", "Saliency", "ShapleyValue", "SIG"]
    attr = {"Attention": ["raw_attention", "attention_rollout", "attention_flow"],
            "DeepLift": ["DeepLift_L2"],
            "InputXGradient": ["InputXGradient_mean", "InputXGradient_L2"],
            "IntegratedGradients": ["IntegratedGradients_mean"],
            "KernelShap": ["ShapleyValue"],
            "Lime": ["Lime"],
            "Occlusion": ["Occlusion"],
            "Saliency": ["Saliency_L2"],
            "ShapleyValue": ["ShapleyValue"],
            "SIG": ["SIG_mean"],
            "decompx": ["DecompX"],}
    for method in methods:
        explanation_path = f"/nethome/yifwang/B-cos-v2/final_results_baseline/perturbation_{dataset}_{model}_train_all/{method}_explanations.json"
        for m in attr[method]:
            save_path = f"visualization_baseline/{dataset}_{model}_{method}_{m}_visualization.html"
            html = print_importance_dataset(explanation_path, m, save_path, no_cls_sep=False, num_examples=-1)
            print(f"Saved {save_path}")
    """
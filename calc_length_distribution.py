# compute the length distribution of the standfordnlp/imdb and fancyzhx/ag_news datasets using bert tokenizer
from transformers import BertTokenizer
from datasets import load_dataset
import numpy as np
import json
import random
import torch
from torch.utils.data import Subset
from tqdm import tqdm

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calc_distribution(dataset, max_length=512):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    lengths = []
    for example in tqdm(dataset):
        text = example['text']
        tokenized_text = tokenizer(text, truncation=True, max_length=max_length)
        lengths.append(len(tokenized_text['input_ids']))
    length_distribution = {}
    length_distribution['total'] = len(lengths)
    length_distribution['mean'] = np.mean(lengths).item()
    length_distribution['median'] = np.median(lengths).item()
    length_distribution['max'] = np.max(lengths).item()
    length_distribution['min'] = np.min(lengths).item()
    length_distribution['std'] = np.std(lengths).item()
    length_distribution['25th percentile'] = np.percentile(lengths, 25).item()
    length_distribution['75th percentile'] = np.percentile(lengths, 75).item()
    return length_distribution


def split_dataset(dataset, ratio):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(ratio * dataset_size))
    np.random.shuffle(indices)

    val_indices, test_indices = indices[:split], indices[split:]

    test_dataset = Subset(dataset, test_indices)
    return test_dataset

if __name__ == "__main__":
    set_random_seed(42)
    """
    imdb = load_dataset('imdb')
    imdb_test = split_dataset(imdb['test'], 0.5)
    imdb_length_distribution = calc_distribution(imdb_test, max_length=512)
    with open('imdb_length_distribution.json', 'w') as f:
        json.dump(imdb_length_distribution, f)
    
    ag_news = load_dataset('ag_news')
    ag_news_test = split_dataset(ag_news['test'], 0.5)
    ag_news_length_distribution = calc_distribution(ag_news_test, max_length=512)
    with open('ag_news_length_distribution.json', 'w') as f:
        json.dump(ag_news_length_distribution, f)
    """
    hatexplain = load_dataset("agvidit1/hateXplain_processed_dataset")
    hatexplain_test = hatexplain['test']
    hatexplain_length_distribution = calc_distribution(hatexplain_test, max_length=512)
    with open('hatexplain_length_distribution.json', 'w') as f:
        json.dump(hatexplain_length_distribution, f)
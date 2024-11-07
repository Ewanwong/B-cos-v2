import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensuring deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_dataset(dataset, split_ratio):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(split_ratio * dataset_size))
    np.random.shuffle(indices)

    val_indices, test_indices = indices[:split], indices[split:]
    dataset = Subset(dataset, test_indices)
    return dataset

def batch_loader(dataset, batch_size, shuffle=False):
    if type(dataset) == dict:
        # get the length of the dataset
        length = len(dataset[list(dataset.keys())[0]])
        indices = list(range(length))
        if shuffle:
            random.shuffle(indices)
        for i in range(0, length, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = {key: [dataset[key][j] for j in batch_indices] for key in dataset.keys()}
            yield batch
    elif type(dataset) == list:
        length = len(dataset)
        indices = list(range(length))
        if shuffle:
            random.shuffle(indices)
        for i in range(0, length, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = [dataset[j] for j in batch_indices]
            yield batch

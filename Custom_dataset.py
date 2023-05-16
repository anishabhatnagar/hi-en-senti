import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

def preprocess_dataset(dataset: Dataset, tokenizer):
    def tokenize_function(data):
        #return tokenizer(data['text'], padding=True, truncation=True)
        return tokenizer(data['text'], padding='max_length', truncation=True, max_length=512)
    return dataset.map(tokenize_function, batched=True)

def read_data(data_dir, language):
    texts = []
    labels = []
    full_set = pd.read_csv(data_dir)
    full_set.replace('', np.nan, inplace=True)
    full_set.dropna(inplace=True)  #subset = ["Hindi"], 
    texts = full_set[language].tolist()
    print(type(texts))
    labels = full_set["Sentiment_mapping"].tolist()
    return texts, labels


class SentiMix_dataset(Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def create_sentimix_dataset(data_dir, language, tokenizer):
    texts, labels = read_data(data_dir, language)
    encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=512)
    #encodings = tokenizer(texts, truncation=True, padding=True)
    dataset = SentiMix_dataset(encodings, labels)
    return dataset


    

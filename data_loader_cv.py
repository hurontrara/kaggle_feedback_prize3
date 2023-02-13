import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer


class TextClassificationDataset(Dataset):

    def __init__(self, texts, labels, config):
        self.texts = texts
        self.labels = labels
        self.config = config

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):

        text = self.texts[item]
        label = self.labels[item]

        return {
            'text' : text,
            'label' : label
        }


class TextClassificationCollator():

    def __init__(self, tokenizer, max_length, with_text=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text

    def __call__(self, samples):
        texts = [s['text'] for s in samples]
        labels = [s['label'] for s in samples]

        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True,
            max_length=self.max_length,
        )

        return_value = {
            'input_ids' : encoding['input_ids'],
            'attention_mask' : encoding['attention_mask'],
            'labels' : torch.tensor(np.array(labels), dtype=torch.float32),
            # 'token_type_ids' : encoding['token_type_ids'],
        }
        if self.with_text:
            return_value['text'] = texts

        return return_value
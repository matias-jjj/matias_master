import torch
import torch.nn.functional as F
from collections import namedtuple
import pandas as pd
import time

class CollationClass:
    def __init__(self, tokenizer, max_length, device):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __call__(self, list_of_sents):
        # this is where tokenization (including padding) is handled

        source_sents = [sentence["source"] for sentence in list_of_sents]
        target_sents = [sentence["target"] for sentence in list_of_sents]

        source_encoding = self.tokenizer(source_sents, 
            padding="longest",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
                ).to(self.device)
        target_encoding = self.tokenizer(target_sents, 
            padding="longest",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
                ).to(self.device)
        # change padding in target ids to accomodate T5-model loss-function
        modified_encodings = []
        for encoding in target_encoding.input_ids:
            modified_encoding = [token if token != 0 else -100 for token in encoding]
            modified_encodings.append(modified_encoding)
        target_encoding.input_ids = torch.tensor(modified_encodings).to(self.device)

        return source_encoding, target_encoding


class T5GecDataset(torch.utils.data.Dataset):
    def __init__(self, data_pandas, args):
        """
        Save sentences as tokenized pandas-series with lists
        """
        self.args = args
        self.source = data_pandas.iloc[:,0] 
        self.target = data_pandas.iloc[:,1]

    def __getitem__(self, index):
        """
        One sentence at the time
        return: RAW sentences in dict
        DataLoaders collate class handles padding
        """
        source_sentence = self.source[index] 
        target_sentence = self.target[index]
        if self.args.casefold:
            source_sentence = source_sentence.lower()
            target_sentence = target_sentence.lower()
        sentences = {"source":source_sentence, "target":target_sentence}
        return sentences

    def __len__(self):
        return len(self.source)
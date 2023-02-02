from typing import List, Dict

from torch.utils.data import Dataset
import re
from utils import Vocab, pad_to_len
import torch

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: tag for tag, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        # the additional 1 is for padding
        return len(self.label_mapping) + 1

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        results = {'tokens': [], 'tags': []}
        for data in samples:
            results['tokens'].append(data['tokens'])
            try:
                results['tags'].append(list(map(lambda x: self.label_mapping[x], data['tags'])))
            except:
                pass
        # in utils.py
        results['tokens'] = torch.LongTensor(self.vocab.encode_batch(results['tokens'], self.max_len))
        try:
            results['tags'] = torch.LongTensor(pad_to_len(results['tags'], self.max_len, 9))
        except:
            pass
        return results
        raise NotImplementedError

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

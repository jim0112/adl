from typing import List, Dict

from torch.utils.data import Dataset
import re
from utils import Vocab
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
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        results = {'text': [], 'intent': [], 'id': []}
        for data in samples:
            results['id'].append(data['id'])
            results['text'].append(re.sub(r"(\w)([^a-zA-Z0-9 ])", r"\1 \2", data['text']).split(" "))
            try:
                results['intent'].append(self.label_mapping[data['intent']])
            except:
                pass
        # in utils.py
        results['text'] = torch.cuda.LongTensor(self.vocab.encode_batch(results['text'], self.max_len))
        try:
            results['intent'] = torch.cuda.LongTensor(results['intent'])
        except:
            pass
        return results
        raise NotImplementedError

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

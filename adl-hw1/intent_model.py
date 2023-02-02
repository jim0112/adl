from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        device
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.device = device
        self.lstm = nn.GRU(input_size=300, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.fc1 = nn.Linear(self.encoder_output_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_class)
        self.activate = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(p=dropout)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        if(self.bidirectional):
            return self.hidden_size * 2
        return self.hidden_size
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        h0 = torch.zeros(self.num_layers * 2, batch.size(0), self.hidden_size).to(self.device)
        #c0 = torch.zeros(self.num_layers * 2, batch.size(0), self.hidden_size).to(self.device)
        batch = self.embed(batch) # to {b, s, f}
        #print(batch.shape)
        out, _ = self.lstm(batch, h0)
        out = self.activate(out)
        # take the last timestep
        out = self.fc1(out[:, -1, :].view(out.size(0), -1))
        out = self.drop(out)
        out = self.activate(out)
        out = self.fc2(out)
        return out
        raise NotImplementedError

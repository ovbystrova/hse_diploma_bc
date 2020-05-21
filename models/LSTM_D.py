import torch.nn as nn
import torch
import math
import configuration as cfg
from utils.helpers import truncated_normal_


class LSTM_D(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_size, emb_pretrained, max_seq_len, padding_idx,
                 weights, gpu=cfg.if_cuda):
        super(LSTM_D, self).__init__()
        self.emb_pretrained = emb_pretrained
        self.embedding = nn.Linear(vocab_size, embed_dim, bias=False)
        # self.embedding.weight.data = embeddings.weight.data # If use pretrained embeddings u
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.rnn = nn.LSTM(input_size=embed_dim,
                           hidden_size=hidden_size,
                           bidirectional=True,
                           batch_first=True,
                           )
        self.fc = nn.Linear(hidden_size * 2 * 2, 2)
        self.weights = weights
        self.init_params()
        self.gpu = gpu

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if self.weights == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif self.weights == 'normal':
                    torch.nn.init.normal_(param, std=stddev)
                elif self.weights == 'truncated_normal':
                    truncated_normal_(param, std=stddev)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, cell) = self.rnn(x)
        hidden = hidden.transpose(0, 1)
        cell = cell.transpose(0, 1)
        hidden = hidden.contiguous().view(hidden.size(0), -1)
        cell = cell.contiguous().view(cell.size(0), -1)
        x = torch.cat([hidden, cell], dim=1).squeeze(1)
        x = self.fc(x)
        return x

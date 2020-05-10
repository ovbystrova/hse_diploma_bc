import torch
import torch.nn as nn
import math


class SimpleD(nn.Module):
    def __init__(self, embed_dim, max_seq_len, vocab_size, weights, dropout=0.25):
        super(SimpleD, self).__init__()

        self.embed_dim = embed_dim
        self.embeddings = nn.Linear(vocab_size, embed_dim, bias=False)
        self.out2logits = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(40, 1)
        self.weights = weights
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if self.weights == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif self.weights == 'normal':
                    torch.nn.init.normal_(param, std=stddev)

    def forward(self, inp):
        """
        Get logits of discriminator
        :param inp: batch_size * seq_len * vocab_size
        :return logits: [batch_size] (1-D tensor)
        """

        emb = self.embeddings(inp)  # batch_size * seq_len * embed_dim
        # print(emb.size())
        logits = self.out2logits(emb).squeeze(1)  # batch_size * seq_leq
        # print(logits.size())
        logits = self.dropout(logits).squeeze(-1)
        # print(logits.size())
        logits = self.fc(logits)
        # print(logits.size())

        return logits  # batch_size
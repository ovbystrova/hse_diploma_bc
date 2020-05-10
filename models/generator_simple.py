import torch.nn.functional as F
import torch.nn as nn


class SimpleG(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_size):
        super(SimpleG, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.rnn = nn.LSTM(input_size=embed_dim,
                           hidden_size=hidden_size,
                           bidirectional=True,
                           batch_first=True,
                           )

        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, batch):
        x = batch.text.T if hasattr(batch, 'text') else batch  #
        x = self.embedding(x)  # (batch_size, sequence_length, embed_dim)
        x, _ = self.rnn(x)  # (batch_size, sequence length, hidden_size)
        x = self.fc(x)  # (batch_size, sequence_length, vocab_size)
        # x = F.gumbel_softmax(x, dim=-1, hard=True)
        return x  # (batch_size, sequence_length, vocab_size)

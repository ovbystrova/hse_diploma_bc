import torch.nn as nn


class BERT_D(nn.Module):
    def __init__(self, bert, vocab_size, embed_dim, gpu):
        super(BERT_D, self).__init__()
        self.bert = bert
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.gpu = gpu
        self.init_bert()
        self.fc = nn.Linear(self.vocab_size, 1)

    def init_bert(self):
        self.bert.distilbert.word_embeddings = nn.Linear(28996, 768, bias=False)  # TODO пересмотреть

    def forward(self, x):  # (batch_size, sequence length)
        x = self.fc(x).squeeze(-1)
        x = self.bert(x)[0]
        return x
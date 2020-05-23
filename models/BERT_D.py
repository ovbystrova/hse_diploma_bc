import torch.nn as nn


class BERT_D(nn.Module):
    def __init__(self, bert, vocab_size, embed_dim, gpu):
        super(BERT_D, self).__init__()
        self.bert = bert
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.gpu = gpu
        self.init_bert()

    def init_bert(self):
        self.bert.distilbert.embeddings = nn.Linear(self.vocab_size, 768)

    def forward(self, x):  # (batch_size, sequence length)
        print(self.bert.distilbert)
        x = self.bert(x)[0]
        return x
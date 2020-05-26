import torch.nn as nn


class BERT_D(nn.Module):
    def __init__(self, bert, vocab_size, freeze_bert=True):
        super(BERT_D, self).__init__()
        self.bert = bert
        self.vocab_size = vocab_size
        self.fc = nn.Linear(self.vocab_size, 768)
        self.freeze_bert = freeze_bert
        self.init_bert()

    def init_bert(self):
        self.bert.resize_token_embeddings(self.vocab_size)
        if self.freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
            for p in self.bert.pre_classifier.parameters():
                p.requires_grad = True
            for p in self.bert.classifier.parameters():
                p.requires_grad = True

    def forward(self, x):  # (batch_size, sequence length)
        x = self.fc(x)
        print(x.size())
        x = self.bert(inputs_embeds=x)[0]
        return x

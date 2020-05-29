# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : config.py
# @Time         : Created at 2019-03-18
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import configuration as cfg
from utils.helpers import truncated_normal_


class LSTMGenerator(nn.Module):

    def __init__(self, embedding_dim,
                 hidden_dim,
                 vocab_size,
                 max_seq_len,
                 padding_idx,
                 weights,
                 batch_size=cfg.BATCH_SIZE,
                 gpu=cfg.if_cuda):
        super(LSTMGenerator, self).__init__()
        self.name = 'vanilla'

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.gpu = gpu
        self.weights = weights
        self.batch_size = batch_size

        self.temperature = 1.0

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm2out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.init_params()

    def forward(self, inp, hidden, need_hidden=False):
        """
        Embeds input and applies LSTM
        :param inp: batch_size * seq_len
        :param hidden: (h, c)
        :param need_hidden: if return hidden, use for sampling
        """
        emb = self.embeddings(inp)  # batch_size * len * embedding_dim
        if len(inp.size()) == 1:
            emb = emb.unsqueeze(1)  # batch_size * 1 * embedding_dim

        out, hidden = self.lstm(emb, hidden)  # out: batch_size * seq_len * hidden_dim
        out = out.contiguous().view(-1, self.hidden_dim)  # out: (batch_size * len) * hidden_dim
        out = self.lstm2out(out)  # (batch_size * seq_len) * vocab_size
        # out = self.temperature * out  # temperature
        pred = self.softmax(out)

        if need_hidden:
            return pred, hidden
        else:
            return pred


    def step(self, inp, hidden):
        """
        RelGAN step forward
        :param inp: [batch_size]
        :param hidden: memory size
        :return: pred, hidden, next_token, next_token_onehot, next_o
            - pred: batch_size * vocab_size, use for adversarial training backward
            - hidden: next hidden
            - next_token: [batch_size], next sentence token
            - next_token_onehot: batch_size * vocab_size, not used yet
            - next_o: batch_size * vocab_size, not used yet
        """
        emb = self.embeddings(inp).unsqueeze(1)
        out, hidden = self.lstm(emb, hidden)
        gumbel_t = self.add_gumbel(self.lstm2out(out.squeeze(1)))
        next_token = torch.argmax(gumbel_t, dim=1).detach()
        # next_token_onehot = F.one_hot(next_token, cfg.vocab_size).float()  # not used yet
        next_token_onehot = None

        pred = F.softmax(gumbel_t * self.temperature, dim=-1)  # batch_size * vocab_size
        # next_o = torch.sum(next_token_onehot * pred, dim=1)  # not used yet
        next_o = None

        return pred, hidden, next_token, next_token_onehot, next_o


    def sample(self, num_samples, batch_size, one_hot=False, start_letter=cfg.START_IDX):
        """
        Sample from RelGAN Generator
        - one_hot: if return pred of RelGAN, used for adversarial training
        :return:
            - all_preds: batch_size * seq_len * vocab_size, only use for a batch
            - samples: all samples
        """
        global all_preds
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long().to(cfg.device)
        if one_hot:
            all_preds = torch.zeros(batch_size, self.max_seq_len, self.vocab_size).to(cfg.device)
        for b in range(num_batch):
            hidden = self.init_hidden(batch_size)
            inp = torch.LongTensor([start_letter] * batch_size).to(cfg.device)

            for i in range(self.max_seq_len):
                pred, hidden, next_token, _, _ = self.step(inp, hidden)
                samples[b * batch_size:(b + 1) * batch_size, i] = next_token
                if one_hot:
                    all_preds[:, i] = pred
                inp = next_token
        samples = samples[:num_samples]  # num_samples * seq_len
        if one_hot:
            return all_preds  # batch_size * seq_len * vocab_size
        return samples.float()

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

    def init_oracle(self):
        for param in self.parameters():
            if param.requires_grad:
                torch.nn.init.normal_(param, mean=0, std=1)

    def init_hidden(self, batch_size=cfg.BATCH_SIZE):
        h = torch.zeros(1, batch_size, self.hidden_dim).to(cfg.device)
        c = torch.zeros(1, batch_size, self.hidden_dim).to(cfg.device)
        return h, c

    @staticmethod
    def add_gumbel(o_t, eps=1e-10):
        """Add o_t by a vector sampled from Gumbel(0,1)"""
        u = torch.zeros(o_t.size()).to(cfg.device)
        u.uniform_(0, 1).to(cfg.device)
        g_t = -torch.log(-torch.log(u + eps) + eps)
        gumbel_t = o_t.to(cfg.device) + g_t
        return gumbel_t

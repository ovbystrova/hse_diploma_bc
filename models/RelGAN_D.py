# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : config.py
# @Time         : Created at 2019-03-18
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.helpers import truncated_normal_
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if_cuda = True if torch.cuda.is_available() else False

dis_filter_sizes = [2, 3, 4, 5]
dis_num_filters = [300, 300, 300, 300]

class RelGAN_D(nn.Module):
    def __init__(self, embed_dim, max_seq_len, num_rep, vocab_size, padding_idx, weights, gpu=if_cuda, dropout=0.25):
        super(RelGAN_D, self).__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.feature_dim = sum(dis_num_filters)
        self.emb_dim_single = int(embed_dim / num_rep)
        self.padding_idx = padding_idx
        self.weights = weights
        self.gpu = gpu

        self.embeddings = nn.Linear(vocab_size, embed_dim, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, self.emb_dim_single), stride=(1, self.emb_dim_single)) for (n, f) in
            zip(dis_num_filters, dis_filter_sizes)])
        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 100)
        self.out2logits = nn.Linear(100, 1)
        self.dropout = nn.Dropout(dropout)
        self.init_params()


    def get_feature(self, inp):
        """
        Get feature vector of given sentences
        :param inp: batch_size * max_seq_len
        :return: batch_size * feature_dim
        """
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # tensor: batch_size * feature_dim
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway

        return pred

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


    def forward(self, inp):
        """
        Get logits of discriminator
        :param inp: batch_size * seq_len * vocab_size
        :return logits: [batch_size * num_rep] (1-D tensor)
        """
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim

        cons = [F.relu(conv(emb)) for conv in self.convs]  # [batch_size * num_filter * (seq_len-k_h+1) * num_rep]
        pools = [F.max_pool2d(con, (con.size(2), 1)).squeeze(2) for con in cons]  # [batch_size * num_filter * num_rep]
        pred = torch.cat(pools, 1)
        pred = pred.permute(0, 2, 1).contiguous().view(-1, self.feature_dim)  # (batch_size * num_rep) * feature_dim
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway

        pred = self.feature2out(self.dropout(pred))
        logits = self.out2logits(pred).squeeze(1)  # [batch_size * num_rep]

        return logits

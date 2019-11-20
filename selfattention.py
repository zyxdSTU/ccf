import torch
from torch import nn
import math
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(config['model']['dropout'])
        self.d_k = config['model']['hiddenSize']
        self.queryLinear = nn.Linear(self.d_k, self.d_k)
        self.keyLinear = nn.Linear(self.d_k, self.d_k)
        self.weightLiner = nn.Linear(self.d_k, 1)

    def forward(self, input, mask):
        #input = batchSize x sequenceLen x embeddingSize

        batchSize, sequenceLen, embeddingSize = input.shape
        query = self.queryLinear(input)
        key = self.keyLinear(input)

        query = query.unsqueeze(2).expand(batchSize, sequenceLen, sequenceLen, embeddingSize)
        key = key.unsqueeze(1).expand(batchSize, sequenceLen, sequenceLen, embeddingSize)

        scores = self.weightLiner(torch.tanh(query + key)).squeeze()

        mask = mask.unsqueeze(-1).expand_as(scores)

        scores.masked_fill(mask, -1e9)

        attn = F.softmax(scores, dim=-1)

        attn = self.dropout(attn)

        return torch.matmul(attn, input)


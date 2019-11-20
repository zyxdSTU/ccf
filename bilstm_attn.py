from data_loader import tagDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tgrange
from tqdm import tqdm
import torch.optim as optim
import sys
from torchcrf import CRF
from transformers import BertModel
from transformers import XLNetModel
from selfattention import SelfAttention
import math
class BiLSTM_ATTN(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['model']['pretrained_model'] == 'XLNet':
            self.pretrainedModel = XLNetModel.from_pretrained(config['model']['xlnet_base_chinese'])
        
        if config['model']['pretrained_model'] == 'Bert':
            self.pretrainedModel = BertModel.from_pretrained(config['model']['bert_base_chinese'])
             
        self.dropout = nn.Dropout(config['model']['dropout'])
        self.lstm = nn.LSTM(input_size=768, hidden_size=768//2, batch_first=True,bidirectional=True)
        self.attention = SelfAttention(config)
        self.fc = nn.Linear(768, len(tagDict))
        weight = torch.Tensor([1, 1, 3, 3, 3]).to(config['DEVICE'])
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def forward(self, batchSentence, batchTag):
        self.lstm.flatten_parameters()
        mask = batchSentence.data.gt(0)
        encodedLayers = self.pretrainedModel(batchSentence, attention_mask=mask.float())[0]
        encodedLayers = self.dropout(encodedLayers)
        h, _ = self.lstm(encodedLayers)
        h = self.attention(h, mask)
        h = self.dropout(h)
        h = self.fc(h)
        activeIndex = (batchTag != 0).view(-1)
        h = h.view(-1, len(tagDict))[activeIndex]
        batchTag = batchTag.view(-1)[activeIndex]
        loss = self.criterion(h, batchTag)
        return loss
    
    def decode(self, batchSentence):
        self.lstm.flatten_parameters()
        mask = batchSentence.data.gt(0)
        encodedLayers = self.pretrainedModel(batchSentence, attention_mask=mask.float())[0]
        h, _ = self.lstm(encodedLayers)
        h = self.attention(h, mask)
        h = self.fc(h)
        h = F.softmax(h, dim=2)
        result, probArr = [], []
        for sentenceEle, hEle in zip(batchSentence, h):
            activeIndex = (sentenceEle != 0)
            hEle = hEle[activeIndex]
            result.append([element.argmax().item() for element in hEle])
            probArr.append(hEle.cpu().numpy().tolist())
        return result, probArr#hELe各部分概率

# RNNS = ['LSTM', 'GRU']

# class Encoder(nn.Module):
#   def __init__(self, embedding_dim, hidden_dim, nlayers=1, dropout=0.,
#                bidirectional=True, rnn_type='GRU'):
#     super(Encoder, self).__init__()
#     self.bidirectional = bidirectional
#     assert rnn_type in RNNS, 'Use one of the following: {}'.format(str(RNNS))
#     rnn_cell = getattr(nn, rnn_type) # fetch constructor from torch.nn, cleaner than if
#     self.rnn = rnn_cell(embedding_dim, hidden_dim, nlayers, 
#                         dropout=dropout, bidirectional=bidirectional)

#   def forward(self, input, hidden=None):
#     return self.rnn(input, hidden)


# class Attention(nn.Module):
#   def __init__(self, query_dim, key_dim, value_dim):
#     super(Attention, self).__init__()
#     self.scale = 1. / math.sqrt(query_dim)

#   def forward(self, query, keys, values):
#     # Query = [BxQ]
#     # Keys = [TxBxK]
#     # Values = [TxBxV]
#     # Outputs = a:[TxB], lin_comb:[BxV]

#     # Here we assume q_dim == k_dim (dot product attention)

#     query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
#     keys = keys.transpose(0,1).transpose(1,2) # [TxBxK] -> [BxKxT]
#     energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
#     energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize

#     values = values.transpose(0,1) # [TxBxV] -> [BxTxV]
#     linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
#     return energy, linear_combination

# class BiLSTM_ATTN(nn.Module):
#     def __init__(self, config):
#         super(BiLSTM_ATTN, self).__init__()
#         self.embedding_dim = 768
#         self.dropout = nn.Dropout(config['model']['dropout'])
#         self.bertModel = BertModel.from_pretrained(config['model']['bert_base_chinese'])
#         self.encoder = Encoder(embedding_dim = self.embedding_dim, hidden_dim= self.embedding_dim // 2)
#         self.attention = Attention(self.embedding_dim, self.embedding_dim, self.embedding_dim)
#         self.decoder = nn.Linear(self.embedding_dim, len(tagDict))
#         weight = torch.Tensor([1, 1, 3, 3, 3]).to(config['DEVICE'])
#         self.criterion = nn.CrossEntropyLoss(weight=weight)

#     def forward(self, batchSentence, batchTag):
#         mask = batchSentence.data.gt(0)
#         encodedLayers = self.bertModel(batchSentence, attention_mask=mask.float())[0]
#         encodedLayers = self.dropout(encodedLayers)
#         outputs, hidden = self.encoder(encodedLayers)
#         print (outputs.shape)
#         if isinstance(hidden, tuple): # LSTM
#             hidden = hidden[1] # take the cell state

#         if self.encoder.bidirectional: # need to concat the last 2 hidden layers
#             hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
#         else:
#             hidden = hidden[-1]

#         energy, linear_combination = self.attention(hidden, outputs, outputs) 
#         print (linear_combination.shape)
#         linear_combination = self.dropout(linear_combination)
#         h = self.decoder(linear_combination)
#         print (h.shape)
#         activeIndex = (batchTag != 0).view(-1)
#         h = h.view(-1, len(tagDict))[activeIndex]
#         batchTag = batchTag.view(-1)[activeIndex]
#         loss = self.criterion(h, batchTag)
#         return loss

#     def decode(self, batchSentence):
#         mask = batchSentence.data.gt(0)
#         encodedLayers = self.bertModel(batchSentence, attention_mask=mask.float())[0]
#         outputs, hidden = self.encoder(encodedLayers)
#         if isinstance(hidden, tuple): # LSTM
#             hidden = hidden[1] # take the cell state

#         if self.encoder.bidirectional: # need to concat the last 2 hidden layers
#             hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
#         else:
#             hidden = hidden[-1]

#         energy, linear_combination = self.attention(hidden, outputs, outputs) 
#         linear_combination = self.dropout(linear_combination)
#         h = self.decoder(linear_combination)
#         h = F.softmax(h, dim=2)
#         result, probArr = [], []
#         for sentenceEle, hEle in zip(batchSentence, h):
#             activeIndex = (sentenceEle != 0)
#             hEle = hEle[activeIndex]
#             result.append([element.argmax().item() for element in hEle])
#             probArr.append(hEle.cpu().numpy().tolist())
#         return result, probArr#hELe各部分概率
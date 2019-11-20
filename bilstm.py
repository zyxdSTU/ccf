#from pytorch_pretrained_bert import BertModel
from data_loader import tagDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
from data_loader import id2tag
from pytorch_pretrained_bert import BertTokenizer
from util import acquireEntity
import sys
from util import generateResult
from transformers import BertModel
from transformers import XLNetModel
from transformers import XLNetTokenizer

class BiLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['model']['pretrained_model'] == 'XLNet':
            self.pretrainedModel = XLNetModel.from_pretrained(config['model']['xlnet_base_chinese'])
            self.tokenizer = XLNetTokenizer.from_pretrained(self.config['model']['xlnet_base_chinese'], do_lower_case=True)

        if config['model']['pretrained_model'] == 'Bert':
            self.pretrainedModel = BertModel.from_pretrained(config['model']['bert_base_chinese'])
            self.tokenizer = BertTokenizer.from_pretrained(config['model']['bert_base_chinese'], do_lower_case=True)

        #for p in self.bertModel.parameters(): p.requires_grad = False
        self.dropout = nn.Dropout(config['model']['dropout'])
        self.lstm = nn.LSTM(input_size=768, hidden_size=768//2, batch_first=True,bidirectional=True)#, num_layers=2,dropout=config['model']['dropout'])  
        #self.layerNorm = nn.LayerNorm(768)
        self.fc = nn.Linear(768, len(tagDict))
        #weight = torch.Tensor([1, 1, 2.5, 2.5, 2.5]).to(config['DEVICE'])
        weight = torch.Tensor([1, 1, 3, 3, 3]).to(config['DEVICE'])
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        #self.criterion = nn.CrossEntropyLoss()

    def forward(self, batchSentence, batchTag):
        #self.lstm.flatten_parameters()
        mask = batchSentence.data.gt(0).float()
        encodedLayers = self.pretrainedModel(batchSentence, attention_mask=mask)[0]
        encodedLayers = self.dropout(encodedLayers)
        h, _ = self.lstm(encodedLayers)
        #h = self.layerNorm(h)
        h = self.dropout(h)
        h = self.fc(h)
        activeIndex = (batchTag != 0).view(-1)
        h = h.view(-1, len(tagDict))[activeIndex]
        batchTag = batchTag.view(-1)[activeIndex]
        loss = self.criterion(h, batchTag)
        return loss

    
    def decode(self, batchSentence):
        #self.lstm.flatten_parameters()

        mask = batchSentence.data.gt(0).float()
        encodedLayers = self.pretrainedModel(batchSentence, attention_mask=mask)[0]
        h, _ = self.lstm(encodedLayers)
        h = self.fc(h)
        h = F.softmax(h, dim=2)
        result, probArr = [], []
        for sentenceEle, hEle in zip(batchSentence, h):
            activeIndex = sentenceEle != 0
            hEle = hEle[activeIndex]
            result.append([element.argmax().item() for element in hEle])
            probArr.append(hEle.cpu().numpy().tolist())
        return result, probArr#hELe各部分概率
        

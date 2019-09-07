from pytorch_pretrained_bert import BertModel
from data_loader import tagDict
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bertModel = BertModel.from_pretrained(config['model']['bert_base_chinese'])
        self.dropout = nn.Dropout(config['model']['dropout'])
        self.lstm = nn.LSTM(input_size=768, hidden_size=768//2, batch_first=True,bidirectional=True)
        self.fc = nn.Linear(768, len(tagDict))
    
    def forward(self, batchSentence):
        if self.training:
            self.bertModel.train()
            self.lstm.flatten_parameters()
            encodedLayers, _ = self.bertModel(batchSentence, output_all_encoded_layers=False)
            lstmFeature,_ = self.lstm(encodedLayers)

            #dropout策略
            #lstmFeature = self.dropout(lstmFeature)

            tagFeature = self.fc(lstmFeature)
            tagScores = F.log_softmax(tagFeature, dim=2)
        else:
            self.bertModel.eval()
            with torch.no_grad():
                self.lstm.flatten_parameters()
                encodedLayers, _ = self.bertModel(batchSentence, output_all_encoded_layers=False)
                lstmFeature, _ = self.lstm(encodedLayers)
                #lstmFeature = lstmFeature * config['model']['dropout'] 
                tagFeature = self.fc(lstmFeature)
                tagScores = F.log_softmax(tagFeature, dim=2)
        return tagScores
        


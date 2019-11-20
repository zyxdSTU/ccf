import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from data_loader import tagDict
from tqdm import tgrange
from tqdm import tqdm
import torch.optim as optim
from data_loader import id2tag
import sys
from transformers import BertModel

class IDCNN(nn.Module):
    def __init__(self, config, layerNumber=4, blockNumber=3,kernel_size=3):
        super(IDCNN, self).__init__()
        self.batchSize = config['model']['batchSize']
        self.dropout = config['model']['dropout']
        self.device = config['DEVICE']
        self.hiddenSize = config['model']['hiddenSize']

        self.bertModel = BertModel.from_pretrained(config['model']['bert_base_chinese'])
        self.linear1 = nn.Linear(768, self.hiddenSize)
        self.idcnn = nn.Sequential()
        self.linear2 = nn.Linear(self.hiddenSize, len(tagDict))
        net = nn.Sequential()
        for i in range(layerNumber):
            dilation = int(math.pow(2, i)) if i+1 < layerNumber else 1
            block = nn.Conv1d(in_channels = self.hiddenSize,
                         out_channels = self.hiddenSize,
                         kernel_size = kernel_size,
                         dilation = dilation,
                         padding = kernel_size // 2 + dilation - 1)
            net.add_module("layer%d"%i, block)
            net.add_module("relu", nn.ReLU(True))

        for i in range(blockNumber):
            self.idcnn.add_module("block%i"%i, net)
            self.idcnn.add_module("dropout", nn.Dropout(self.dropout))

        weight = torch.Tensor([1, 1, 3, 3, 3]).to(config['DEVICE'])
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def forward(self, batchSentence, batchTag):
        #字符嵌入
        attention_mask = batchSentence.data.gt(0).float()
        embeddings = self.bertModel(batchSentence, attention_mask=attention_mask)[0]
        embeddings = nn.Dropout(self.dropout)(embeddings)
        input = self.linear1(embeddings)
        input = input.permute(0, 2, 1)
        output = self.idcnn(input).permute(0, 2, 1)
        output = self.linear2(output)
        activeIndex = (batchTag != 0).view(-1)
        output = output.view(-1, len(tagDict))[activeIndex]
        batchTag = batchTag.view(-1)[activeIndex]
        loss = self.criterion(output, batchTag)
        return loss

    def decode(self, batchSentence):
        #字符嵌入
        attention_mask = batchSentence.data.gt(0).float()
        embeddings = self.bertModel(batchSentence, attention_mask=attention_mask)[0]
        input = self.linear1(embeddings)
        input = input.permute(0, 2, 1)
        output = self.idcnn(input).permute(0, 2, 1)
        output = self.linear2(output)
        output = F.softmax(output, dim=2)
        result, probArr = [], []
        for sentenceEle, hEle in zip(batchSentence, output):
            activeIndex = (sentenceEle != 0)
            hEle = hEle[activeIndex]
            result.append([element.argmax().item() for element in hEle])
            probArr.append(hEle.cpu().numpy().tolist())
        return result, probArr
from pytorch_pretrained_bert import BertModel
from data_loader import tagDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tgrange
from tqdm import tqdm
from seqeval.metrics import f1_score, accuracy_score, classification_report
from torch.autograd import Variable
import math, copy, time
from tqdm import tgrange
from tqdm import tqdm
import torch.optim as optim
from seqeval.metrics import f1_score, accuracy_score, classification_report
from data_loader import id2tag
from pytorch_pretrained_bert import BertTokenizer
from data_util import acquireEntity
from data_util import f2_score
import sys

class Transformer_CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.batchSize = config['model']['batchSize']
        self.dropout = nn.Dropout(config['model']['dropout'])
        self.device = config['DEVICE']
        #选取的特征数量
        self.featureLen = config['model']['featureLen']
        self.hiddenSize = config['model']['hiddenSize']
        self.embeddingSize = 768

        self.positionEncoding = PositionalEncoding(self.embeddingSize, dropout = 0.1)
        self.bertModel = BertModel.from_pretrained(config['model']['bert_base_chinese'])

        self.layer = nn.TransformerEncoderLayer(d_model = self.embeddingSize, nhead = 4)

        self.encoder = nn.TransformerEncoder(self.layer, num_layers=2)

        self.cnnArr = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=self.hiddenSize//self.featureLen, kernel_size=(i, self.embeddingSize))
            for i in range(2, 2+ self.featureLen)])

        self.fc = nn.Linear(self.hiddenSize, len(tagDict))

    def forward(self, batchSentence):
        #字符嵌入层
        embeddings, _ = self.bertModel(batchSentence, output_all_encoded_layers=False)
        embeddings = self.positionEncoding(embeddings)

        #Transformer层
        mask = batchSentence == 0
        embeddings = embeddings.permute(1, 0, 2)
        embeddings = self.encoder(embeddings, src_key_padding_mask=mask)
        embeddings = embeddings.permute(1, 0, 2)

        result = []
        for index, cnn in enumerate(self.cnnArr):
            #左、右边padding
            size = index + 2
            if size % 2 != 0:
                paddingLef = paddingRig = (size - 1) // 2
            else:
                paddingLef, paddingRig = size // 2 , size // 2 -1
            
            paddingLef = torch.zeros((embeddings.size()[0], paddingLef, self.embeddingSize)).to(self.device)
            paddingRig = torch.zeros((embeddings.size()[0],  paddingRig, self.embeddingSize)).to(self.device)
            inputData = torch.cat((paddingLef,embeddings, paddingRig), 1)
            inputData = inputData.unsqueeze(1)
            outputData = cnn(inputData)
            outputData = outputData.squeeze(3).transpose(1, 2)
            result.append(outputData)
        
        result = torch.cat(result, 2)
        result = self.dropout(result)
        result = self.fc(result)
        #print (result.shape)
        result = F.log_softmax(result, dim=2)
        return result

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                                -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                            requires_grad=False)
        return self.dropout(x)

def transformer_cnn_train(net, trainIter, validIter, config):
    DEVICE = config['DEVICE']
    modelSavePath = config['modelSavePath']
    validLenPath = config['data']['validLenPath']
    epochNum = config['model']['epochNum']
    learningRate = config['model']['learningRate']
    earlyStop = config['model']['earlyStop']
    
    #设置不同的学习率
    optimizer = optim.Adam(net.parameters(), lr=learningRate)
    optimizer = optim.SGD([{'params': net.bertModel.parameters(), 'lr': learningRate /5},
                       {'params': net.encoder.parameters(), 'lr': learningRate / 5},
                       {'params': net.cnnArr.parameters()}, 
                       {'params': net.fc.parameters()}], lr=learningRate)

    criterion = nn.NLLLoss()
    earlyNumber, beforeLoss, maxScore = 0, sys.maxsize, -1
    for epoch in range(epochNum):
        print ('第%d次迭代\n' % (epoch+1))

        #训练
        net.train()
        trainLoss, number = 0, 0
        for batchSentence, batchTag, lenList, _ in tqdm(trainIter):
            batchSentence = batchSentence.to(DEVICE)
            batchTag = batchTag.to(DEVICE)
            net.zero_grad()
            tagScores = net(batchSentence)
            loss = 0
            for index, element in enumerate(lenList):
                tagScore = tagScores[index][:element]
                tag = batchTag[index][:element]
                loss +=  criterion(tagScore, tag)
            loss.backward()
            optimizer.step()
            trainLoss += loss.item(); number += 1
        trainLoss = trainLoss / number

        #验证
        net.eval()
        validLoss, number = 0, 0
        yTrue, yPre, ySentence = [], [], []
        net.eval()
        with torch.no_grad():
            for batchSentence, batchTag, lenList, originSentence in tqdm(validIter):
                batchSentence = batchSentence.to(DEVICE)
                batchTag = batchTag.to(DEVICE)

                tagScores  = net(batchSentence); loss = 0
                #print (tagScores)
                ySentence.extend(originSentence)
                for index, element in enumerate(lenList):
                    tagScore = tagScores[index][:element]
                    tag = batchTag[index][:element]
                    #print (tagScore.shape, tag.shape)
                    loss +=  criterion(tagScore, tag)
                    sentence = batchSentence[index][:element]
                    yTrue.append(tag.cpu().numpy().tolist())
                    yPre.append([element.argmax().item() for element in tagScore])

                validLoss += loss.item(); number += 1

        yTrue2tag = [[id2tag[element2] for element2 in element1] for element1 in yTrue]
        yPre2tag = [[id2tag[element2] for element2 in element1] for element1 in yPre]

        assert len(yTrue2tag) == len(yPre2tag); assert len(ySentence) == len(yTrue2tag)

        f2Score = f2_score(y_true=yTrue2tag, y_pred=yPre2tag, y_Sentence=ySentence, validLenPath=validLenPath)
        f1Score = f1_score(y_true=yTrue2tag, y_pred=yPre2tag)
        validLoss = validLoss / number

        
        if validLoss <  beforeLoss:
            beforeLoss = validLoss
            torch.save(net.state_dict(), modelSavePath)

        print ('训练损失为: %f\n' % trainLoss)
        print ('验证损失为: %f / %f\n' % (validLoss, beforeLoss))
        print ('f1_Score: %f f2_Score: %f\n' % (f1Score, f2Score))
        #print ('f1_Score: %f \n' % f1Score)

        #早停机制
        if validLoss >  beforeLoss:
            earlyNumber += 1
            print('earyStop: %d / %d\n' % (earlyNumber, earlyStop))
        else:
            earlyNumber = 0
        if earlyNumber >= earlyStop: break


def transformer_cnn_test(net, testIter, config):
    testLenPath = config['data']['testLenPath']
    submitDataPath = config['submitDataPath']
    batchSize = config['model']['batchSize']
    DEVICE = config['DEVICE']

    submitData = open(submitDataPath, 'w', encoding='utf-8', errors='ignore')
    testLen = open(testLenPath, 'r', encoding='utf-8', errors='ignore')
    submitData.write('id,unknownEntities\n')
    sentenceArr, tagArr = [], []
    with torch.no_grad():
        for batchSentence, batchOriginSentence, lenList in tqdm(testIter):
            batchSentence = batchSentence.to(DEVICE)
            tagScores = net(batchSentence)
            for index, element in enumerate(lenList):
                tagScore = tagScores[index][:element]
                tagArr.append([element.argmax().item() for element in tagScore])
            sentenceArr.extend(batchOriginSentence)

    assert len(sentenceArr) == len(tagArr)

    #id转标识
    tagArr =[[id2tag[element2] for element2 in element1]for element1 in tagArr]

    lenList = []
    start, end = 0, 0
    for line in testLen.readlines():
        id, length = line.strip('\n').split('\t')[0], int(line.strip('\n').split('\t')[1])
        sentenceElement, tagElement = sentenceArr[start:start+length], tagArr[start:start+length]
        start += length

        entityArr = acquireEntity(sentenceElement, tagElement)

        def filter_word(w):
            for wbad in ['？','《','🔺','️?','!','#','%','%','，','Ⅲ','》','丨','、','）','（','​',
                    '👍','。','😎','/','】','-','⚠️','：','✅','㊙️','“',')','(','！','🔥',',','.','——', '“', '”', '！', ' ']:
                if wbad in w:
                    return ''
            return w

        #过滤一些无用实体
        entityArr = [entity for entity in entityArr if filter_word(entity) != '' and len(entity) > 1]


        submitData.write('%s,%s\n' % (id, ';'.join(entityArr)))

    submitData.close(); testLen.close()
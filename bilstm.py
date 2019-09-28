from pytorch_pretrained_bert import BertModel
from data_loader import tagDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tgrange
from tqdm import tqdm
import torch.optim as optim
from seqeval.metrics import f1_score, accuracy_score, classification_report
from data_loader import id2tag
from pytorch_pretrained_bert import BertTokenizer
from data_util import acquireEntity
from data_util import f2_score
import sys
from crf import crf
from crf import START_TAG, STOP_TAG, PAD

class BiLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bertModel = BertModel.from_pretrained(config['model']['bert_base_chinese'])
        self.lstmDropout = nn.Dropout(config['model']['lstmDropout'])
        self.embeddingDropout = nn.Dropout(config['model']['embeddingDropout'])
        self.lstm = nn.LSTM(input_size=768, hidden_size=768//2, batch_first=True,bidirectional=True)  
        self.fc = nn.Linear(768, len(tagDict))

    def forward(self, batchSentence):
        encodedLayers, _ = self.bertModel(batchSentence, output_all_encoded_layers=False)
        encodedLayers = self.embeddingDropout(encodedLayers)
        h, _ = self.lstm(encodedLayers)
        h = self.lstmDropout(h)
        h = self.fc(h)
        h = F.log_softmax(h, dim=2)
        return h
    

def bilstm_train(net, trainIter, validIter, config):
    DEVICE = config['DEVICE']
    modelSavePath = config['modelSavePath']
    validLenPath = config['data']['validLenPath']
    epochNum = config['model']['epochNum']
    learningRate = config['model']['learningRate']
    earlyStop = config['model']['earlyStop']
    optimizer = optim.SGD([{'params': net.bertModel.parameters(), 'lr': learningRate /5},
                       {'params': net.lstm.parameters()}, 
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

            net.zero_grad(); loss = 0
            tagScores  = net(batchSentence)
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
        with torch.no_grad():
            for batchSentence, batchTag, lenList, originSentence in tqdm(validIter):
                batchSentence = batchSentence.to(DEVICE)
                batchTag = batchTag.to(DEVICE)
                tagScores  = net(batchSentence)

                ySentence.extend(originSentence); loss = 0
                for index, element in enumerate(lenList):
                    tagScore = tagScores[index][:element]
                    tag = batchTag[index][:element]
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
        print ('f1_Score: %f f2_Score %f\n' % (f1Score, f2Score))

        #早停机制
        if validLoss >  beforeLoss:
            earlyNumber += 1
            print('earyStop: %d / %d\n' % (earlyNumber, earlyStop))
        else:
            earlyNumber = 0
        if earlyNumber >= earlyStop: break


def bilstm_test(net, testIter, config):
    testLenPath = config['data']['testLenPath']
    submitDataPath = config['submitDataPath']
    batchSize = config['model']['batchSize']
    DEVICE = config['DEVICE']

    submitData = open(submitDataPath, 'w', encoding='utf-8', errors='ignore')
    testLen = open(testLenPath, 'r', encoding='utf-8', errors='ignore')
    submitData.write('id,unknownEntities\n')
    sentenceArr, tagArr = [], []
    with torch.no_grad():
        for batchSentence, batchOriginSentence,  lenList in tqdm(testIter):
            batchSentence = batchSentence.to(DEVICE)
            tagScores = net(batchSentence)
            for index, element in enumerate(lenList):
                tagScore = tagScores[index][:element]
                tagArr.append([element.argmax().item() for element in tagScore])
            sentenceArr.extend(batchOriginSentence)

    #id转标识
    tagArr =[[id2tag[element2] for element2 in element1]for element1 in tagArr]
    assert len(sentenceArr) == len(tagArr)
    lenList = []
    start, end = 0, 0
    for line in testLen.readlines():
        id, length = line.strip('\n').split('\t')[0], int(line.strip('\n').split('\t')[1])
        sentenceElement, tagElement = sentenceArr[start:start+length], tagArr[start:start+length]
        start += length

        entityArr = acquireEntity(sentenceElement, tagElement, method='BIOES')

        tokenizer = BertTokenizer.from_pretrained(config['model']['bert_base_chinese'], do_lower_case=True)

        def filter_word(w):
            import string
            errorList = ['？','《','🔺','️?','!','#','%','%','，','Ⅲ','》','丨','、','）','（','​',
                '👍','。','😎','/','】','-','⚠️','：','✅','㊙️','“',')','(','！','🔥',',','.','——', 
                '“', '”', '！', '…', '❶ ','❗️️', '❸','💰','✊', '﻿', '💥', '🌺', '🍀', '➕','❾',
                '😘', '⬇️', '🏦', '☟', '👆', '', '💪', '💡', '🌏', '💚', '💙', '💛']

            for word in w:
                if word not in  string.ascii_letters and word not in tokenizer.vocab.keys():
                    return ''
                if word == ' ' or word in errorList: return ''
            return w

        #过滤一些无用实体
        entityArr = [entity for entity in entityArr if filter_word(entity) != '' and len(entity) > 1]

        submitData.write('%s,%s\n' % (id, ';'.join(entityArr)))

    submitData.close(); testLen.close()

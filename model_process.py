from zhon.hanzi import punctuation
import pandas as pd
from data_loader import id2tag
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from tqdm import tqdm
from util import generateResult
from util import f1_score
import copy
from transformers import AdamW
from util import acquireEntity

def train(net, trainIter, validIter, config):
    DEVICE = config['DEVICE']
    modelSavePath = config['modelSavePath']
    epochNum = config['model']['epochNum']
    learningRate = config['model']['learningRate']
    earlyStop = config['model']['earlyStop']

    #权重初始化
    for name, value in net.named_parameters():
        if 'pretrainedModel' not in name:
            if value.dim() > 1: nn.init.xavier_uniform_(value)

    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    # bert_param_no = [value for name, value in net.named_parameters() if name in no_decay and 'bertModel' in name]
    # bert_param_yes = [value for name, value in net.named_parameters() if name not in no_decay and 'bertModel' in name]

    # other_param_no = [value for name, value in net.named_parameters() if name in no_decay and 'bertModel' not in name]
    # other_param_yes = [value for name, value in net.named_parameters() if name not in no_decay and 'bertModel' not in name]

    
    # optimizer_grouped_parameters = [
    #     {'params': bert_param_yes, 'weight_decay': 0.01, 'lr': learningRate},
    #     {'params': bert_param_no, 'weight_decay': 0.0, 'lr': learningRate}, 
    #     {'params': other_param_yes, 'weight_decay': 0.01, 'lr': 0.001},
    #     {'params': other_param_no, 'weight_decay': 0.0, 'lr': 0.001}]

    bert_params = [value for name, value in net.named_parameters() if 'pretrainedModel' in name]
    other_params = [value for name, value in net.named_parameters() if 'pretrainedModel' not in name]

    params = [{'params':bert_params, 'lr': 5e-5}, 
    {'params':other_params, 'lr':learningRate}]

    optimizer = AdamW(params, eps=1e-8)

    earlyNumber, beforeLoss = 0, sys.maxsize
    trainLossSave, validLossSave, f1ScoreSave, accurateSave, recallSave = 0, 0, 0, 0, 0
    

    for epoch in range(epochNum):
        print ('第%d次迭代\n' % (epoch+1))
        #训练
        net.train()
        trainLoss, number = 0, 0
        for batchSentence, batchTag, _, _ in tqdm(trainIter):
            batchSentence = batchSentence.to(DEVICE)
            batchTag = batchTag.to(DEVICE)
            net.zero_grad()
            loss  = net(batchSentence, batchTag)
            #多卡训练
            if torch.cuda.device_count() > 1: loss = loss.mean()

            loss.backward()

            #梯度裁剪
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            
            optimizer.step()
            trainLoss += loss.item(); number += 1
        trainLoss = trainLoss / number
        
        #验证
        net.eval()
        validLoss, number = 0, 0
        yTrue, yPre, ySentence, probArr = [], [], [], []
        with torch.no_grad():
            for batchSentence, batchTag, lenList, originSentence in tqdm(validIter):
                batchSentence = batchSentence.to(DEVICE)
                batchTag = batchTag.to(DEVICE)
                loss  = net(batchSentence, batchTag)
                #多卡训练
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()
                    tagPre, prob = net.module.decode(batchSentence)
                else: tagPre, prob = net.decode(batchSentence)
                tagTrue = [element[:length] for element, length in zip(batchTag.cpu().numpy(), lenList)]
                yTrue.extend(tagTrue); yPre.extend(tagPre); ySentence.extend(originSentence)
                probArr.extend(prob)
                validLoss += loss.item(); number += 1

        yTrue2tag = [[id2tag[element2] for element2 in element1] for element1 in yTrue]
        yPre2tag = [[id2tag[element2] for element2 in element1] for element1 in yPre]

        assert len(yTrue2tag) == len(yPre2tag); assert len(ySentence) == len(yTrue2tag)

        f1Score, accurate, recall = f1_score(y_true=yTrue2tag, y_pred=yPre2tag)

        validLoss = validLoss / number

        print ('训练损失为: %f\n' % trainLoss)
        print ('验证损失为: %f / %f\n' % (validLoss, beforeLoss))
        print ('f1_Score、accurate、recall: %f、%f、%f\n' % (f1Score, accurate, recall))

        if validLoss <  beforeLoss:
            beforeLoss = validLoss
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), modelSavePath)
            else: torch.save(net.state_dict(), modelSavePath)
            trainLossSave, validLossSave = trainLoss, validLoss
            f1ScoreSave, accurateSave, recallSave = f1Score, accurate, recall
            
            if 'validResultPath' in config.keys():
                path = config['validResultPath']
                f = open(path, 'w', encoding='utf-8', errors='ignore')
                for sentence, prob in zip(ySentence, probArr):
                    for sentenceEle, probEle in zip(sentence, prob):
                        probEle = '\t'.join([str(element) for element in probEle])
                        f.write('%s\t%s\n' %(sentenceEle, probEle))
                    f.write('\n')
                f.close()


        #早停机制
        if validLoss >  beforeLoss:
            earlyNumber += 1
            print('earyStop: %d / %d\n' % (earlyNumber, earlyStop))
        else:
            earlyNumber = 0
        if earlyNumber >= earlyStop: 
            break
    
    #计算验证集中的实际效果
    
    ###临时###
    f = open('temp.txt', 'w', encoding='utf-8', errors='ignore')
    for sentence, trueTag, preTag in zip(ySentence, yTrue2tag, yPre2tag):
        trueEntity = '@'.join(acquireEntity([sentence], [trueTag], method='BIOES'))
        preEntity = '@'.join(acquireEntity([sentence], [preTag], method='BIOES'))

        if trueEntity != preEntity:
            f.write(''.join(sentence) + '\n')
            f.write('True：' + trueEntity + '\n')
            f.write('Pre：' +  preEntity + '\n')
    f.close()

    return trainLossSave, validLossSave, f1ScoreSave, accurateSave, recallSave


def test(net, testIter, config):
    DEVICE = config['DEVICE']
    sentenceArr, tagArr, probArr = [], [], []
    with torch.no_grad():
        for batchSentence, batchOriginSentence, _ in tqdm(testIter):
            batchSentence = batchSentence.to(DEVICE)
            if torch.cuda.device_count() > 1:
                tagPre, prob = net.module.decode(batchSentence)
            else: tagPre, prob = net.decode(batchSentence)
            tagArr.extend(tagPre)
            sentenceArr.extend(batchOriginSentence)
            probArr.extend(prob)
    tagArr =[[id2tag[element2] for element2 in element1]for element1 in tagArr]

    
    #保存中间结果
    if 'resultPath'  in config.keys():
        path = config['resultPath']
        f = open(path, 'w', encoding='utf-8', errors='ignore')
        for sentence, prob in zip(sentenceArr, probArr):
            for sentenceEle, probEle in zip(sentence, prob):
                probEle = '\t'.join([str(element) for element in probEle])
                f.write('%s\t%s\n' %(sentenceEle, probEle))
            f.write('\n')
        f.close()

    disappear1, disappear2 = generateResult(sentenceArr, tagArr, config)

    return disappear1, disappear2

def valid(net, validIter, config):
    def filter_word(w):
        import string
        for word in w:
            if word in ['？','《','🔺','!','#','%','，','Ⅲ','》','丨','、','​', '…',
                    '👍','。','😎','/','】','-','⚠️','：','✅','㊙️','！','🔥',',',
                    '.','——', '“', '”', '！', ' ']:
                return ''
            if word in ['(', ')', '（', '）', '?']: continue
            if word in punctuation + string.punctuation: return ''
        return w
    DEVICE = config['DEVICE']
    sentenceArr, tagArr, probArr = [], [], []
    with torch.no_grad():
        for batchSentence, batchTag, lenList, batchOriginSentence in tqdm(validIter):
            batchSentence = batchSentence.to(DEVICE)
            if torch.cuda.device_count() > 1:
                tagPre, prob = net.module.decode(batchSentence)
            else: tagPre, prob = net.decode(batchSentence)
            tagArr.extend(tagPre)
            sentenceArr.extend(batchOriginSentence)
            probArr.extend(prob)
    tagArr =[[id2tag[element2] for element2 in element1]for element1 in tagArr]
    
    lenPath, comparePath = config['lenPath'], config['comparePath']
    validLen = open(lenPath, 'r', encoding='utf-8', errors='ignore')
    compare = open(comparePath, 'w', encoding='utf-8', errors='ignore')
    lenList, start = [], 0
    TP, FP, FN = 0, 0, 0
    for line in validLen.readlines():
        id, length, trueEntityArr = line.strip('\n').split('\t')[0], int(line.strip('\n').split('\t')[1]), line.strip('\n').split('\t')[2]
        trueEntityArr = trueEntityArr.split(';')
        sentenceElement, tagElement = sentenceArr[start:start+length], tagArr[start:start+length]
        start += length

        preEntityArr = acquireEntity(sentenceElement, tagElement)
        #过滤无用实体
        preEntityArr = [entity for entity in preEntityArr if filter_word(entity) != '']

        compare.write(id + '\t' + ';'.join(trueEntityArr) + '\t' + ';'.join(preEntityArr) + '\n')
        TPE = 0
        for element in preEntityArr:
            if element in trueEntityArr: TPE += 1
           
        FPE = len(preEntityArr) - TPE
        FNE = len(trueEntityArr) - TPE

        TP += TPE; FP += FPE; FN += FNE

    validLen.close(); compare.close()

    if TP+FP == 0 or TP+FN == 0: return 0, 0, 0

    MicroP = TP / (TP + FP); MicroR = TP / (TP + FN)
    
    if MicroP + MicroR == 0: return 0, 0, 0

    MicroF = 2 * MicroP * MicroR / (MicroP + MicroR)

    print('validF1Score %f, validAccurate %f, validRecall %f\n' %(MicroF, MicroP, MicroR))

    return MicroF, MicroP, MicroR
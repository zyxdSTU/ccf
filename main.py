import yaml
from optparse import OptionParser
from data_loader import NERDataset
from data_loader import NERTestDataset
from data_loader import pad
from data_loader import testPad
from torch.utils import data
from model import Net
from tqdm import tgrange
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from data_loader import tagDict
import torch
import sys
import csv
from seqeval.metrics import f1_score, accuracy_score, classification_report
from data_loader import id2tag
from pytorch_pretrained_bert import BertTokenizer
from data_util import acquireEntity

def train(config):
    trainDataPath = config['data']['trainDataPath']
    validDataPath = config['data']['validDataPath']
    modelSavePath = config['data']['modelSavePath']


    batchSize = config['model']['batchSize']
    epochNum = config['model']['epochNum']
    earlyStop = config['model']['earlyStop']
    learningRate = config['model']['learningRate']

    #GPU/CPU
    DEVICE = config['DEVICE']

    trianDataset = NERDataset(trainDataPath, config) 
    validDataset = NERDataset(validDataPath, config)

    trainIter = data.DataLoader(dataset = trianDataset,
                                 batch_size = batchSize,
                                 shuffle = False,
                                 num_workers = 4,
                                 collate_fn = pad)

    validIter = data.DataLoader(dataset = validDataset,
                                 batch_size = batchSize,
                                 shuffle = False,
                                 num_workers = 4,
                                 collate_fn = pad)

    net = Net(config)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    net = net.to(DEVICE)

    lossFunction = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=learningRate)
    earlyNumber, beforeLoss, maxScore = 0, sys.maxsize, -1

    for epoch in range(epochNum):
        print ('第%d次迭代' % (epoch+1))
        trainLoss = trainFun(net, trainIter, optimizer=optimizer, criterion=lossFunction, DEVICE=DEVICE)
        validLoss, f1Score = evalFun(net,validIter,criterion=lossFunction, DEVICE=DEVICE)

        print ('训练损失为: %f' % trainLoss)

        if validLoss < beforeLoss:
            beforeLoss = validLoss
            torch.save(net.state_dict(), modelSavePath)

        print ('验证损失为:%f / %f   f1Score:%f' % (validLoss, beforeLoss, f1Score))

        if validLoss > beforeLoss:
            earlyNumber += 1
            print('earyStop: %d/%d' % (earlyNumber, earlyStop))
        else:
            earlyNumber = 0
        if earlyNumber >= earlyStop: break
        print ('\n')
    
    

def trainFun(net, iterData, optimizer, criterion, DEVICE):
    net.train()
    totalLoss, number = 0, 0
    for batchSentence, batchTag, lenList in tqdm(iterData):
        batchSentence = batchSentence.to(DEVICE)
        batchTag = batchTag.to(DEVICE)
        net.zero_grad()

        loss  = net(batchSentence, batchTag)

        #多卡训练
        if torch.cuda.device_count() > 1:
            loss = loss.mean()

        loss.backward()
        optimizer.step()
        totalLoss += loss.item(); number += 1
    return totalLoss / number

def evalFun(net, iterData, criterion, DEVICE):
    net.eval()
    totalLoss, number = 0, 0
    yTrue, yPre, ySentence = [], [], []
    with torch.no_grad():
        for batchSentence, batchTag, lenList in tqdm(iterData):
            batchSentence = batchSentence.to(DEVICE)
            batchTag = batchTag.to(DEVICE)
            
            loss  = net(batchSentence, batchTag)

            #多卡训练
            if torch.cuda.device_count() > 1:
                loss = loss.mean()

            if torch.cuda.device_count() > 1:
                tagPre = net.module.decode(batchSentence)
            else: tagPre = net.decode(batchSentence)

            tagTrue = [element[:length] for element, length in zip(batchTag.cpu().numpy(), lenList)]
            yTrue.extend(tagTrue); yPre.extend(tagPre)
            totalLoss += loss.item(); number += 1

    yTrue2tag = [[id2tag[element2] for element2 in element1] for element1 in yTrue]
    yPre2tag = [[id2tag[element2] for element2 in element1] for element1 in yPre]
    f1Score = f1_score(y_true=yTrue2tag, y_pred=yPre2tag)
        
    return totalLoss / number, f1Score

def test(config):
    modelSavePath = config['data']['modelSavePath']
    testDataPath = config['data']['testDataPath']
    testLenPath = config['data']['testLenPath']
    submitDataPath = config['data']['submitDataPath']
    batchSize = config['model']['batchSize']
    #GPU/CPU
    DEVICE = config['DEVICE']

    #加载模型
    net = Net(config)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.load_state_dict(torch.load(modelSavePath))
    net = net.to(DEVICE)

    #加载数据
    testDataset = NERTestDataset(testDataPath, config)

    testIter = data.DataLoader(dataset = testDataset,
                                 batch_size = batchSize,
                                 shuffle = False,
                                 num_workers = 4,
                                 collate_fn = testPad)

    submitData = open(submitDataPath, 'w', encoding='utf-8', errors='ignore')
    testLen = open(testLenPath, 'r', encoding='utf-8', errors='ignore')
    
    sentenceArr, tagArr = [], []
    with torch.no_grad():
        for batchSentence, batchOriginSentence in tqdm(testIter):
            batchSentence = batchSentence.to(DEVICE)

            if torch.cuda.device_count() > 1:
                tagPre = net.module.decode(batchSentence)
            else: tagPre = net.decode(batchSentence)

            tagArr.extend(tagPre)
            sentenceArr.extend(batchOriginSentence)

    #id转标识
    tagArr =[[id2tag[element2] for element2 in element1]for element1 in tagArr]

    lenList = []
    start, end = 0, 0
    for line in testLen.readlines():
        id, length = line.strip('\n').split('\t')[0], int(line.strip('\n').split('\t')[1])
        sentenceElement, tagElement = sentenceArr[start:start+length], tagArr[start:start+length]
        start += length

        entityArr = acquireEntity(sentenceElement, tagElement)

        print (entityArr)
            
        def filter_word(w):
            for wbad in ['？','《','🔺','️?','!','#','%','%','，','Ⅲ','》','丨','、','）','（','​',
                    '👍','。','😎','/','】','-','⚠️','：','✅','㊙️','“',')','(','！','🔥',',','.','——', '“', '”', '！', ' ']:
                if wbad in w:
                    return ''
            return w

        #过滤一些无用实体
        entityArr = [entity for entity in entityArr if filter_word(entity) != '' and len(entity) > 1]

        if len(entityArr) == 0: entityArr = ['FUCK']

        submitData.write('%s,%s\n' % (id, ';'.join(entityArr)))

    submitData.close(); testLen.close()


if __name__ == "__main__":
    optParser = OptionParser()
    optParser.add_option('--train',action = 'store_true', dest='train')
    optParser.add_option('--test',action = 'store_true', dest='test')

    f = open('./config.yml', encoding='utf-8', errors='ignore')
    config = yaml.load(f)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print (DEVICE)
    config['DEVICE'] = DEVICE
    f.close()
    option , args = optParser.parse_args()

    #if option.train == True:
    train(config)
        
    #if option.test == True:
    test(config)

        

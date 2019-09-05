import yaml
from optparse import OptionParser
from data_loader import NERDataset
from data_loader import pad
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
from data_util import dispose
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

        if f1Score > maxScore:
            maxScore = f1Score
            torch.save(net.state_dict(), modelSavePath)

        print ('训练损失为: %f' % trainLoss)
        print ('验证损失为:%f   f1Score:%f / %f' % (validLoss, f1Score, maxScore))

        if f1Score < maxScore:
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
        tagScores  = net(batchSentence)

        loss = 0
        for index, element in enumerate(lenList):
            tagScore = tagScores[index][:element]
            tag = batchTag[index][:element]
            loss +=  criterion(tagScore, tag)

        loss.backward()
        optimizer.step()
        totalLoss += loss.item(); number += 1
    return totalLoss / number

def evalFun(net, iterData, criterion, DEVICE):
    net.eval()
    totalLoss, number = 0, 0
    yTrue, yPre, ySentence = [], [], []
    for batchSentence, batchTag, lenList in tqdm(iterData):
        batchSentence = batchSentence.to(DEVICE)
        batchTag = batchTag.to(DEVICE)
        net.zero_grad()
        tagScores  = net(batchSentence)

        loss = 0
        for index, element in enumerate(lenList):
            tagScore = tagScores[index][:element]
            tag = batchTag[index][:element]
            loss +=  criterion(tagScore, tag)
            yTrue.append(tag.cpu().numpy().tolist())
            yPre.append([element.argmax().item() for element in tagScore])

        totalLoss += loss.item(); number += 1

    yTrue2tag = [[id2tag[element2] for element2 in element1] for element1 in yTrue]
    yPre2tag = [[id2tag[element2] for element2 in element1] for element1 in yPre]
    f1Score = f1_score(y_true=yTrue2tag, y_pred=yPre2tag)
        
    return totalLoss / number, f1Score

def test(config):
    modelSavePath = config['data']['modelSavePath']
    testDataPath = config['data']['testDataPath']
    submitDataPath = config['data']['submitDataPath']
    #GPU/CPU
    DEVICE = config['DEVICE']

    #加载模型
    net = Net(config)
    net.load_state_dict(torch.load(modelSavePath))

    testData = open(testDataPath, 'r', encoding='utf-8', errors='ignore')
    submitData = open(testDataPath, 'r', encoding='utf-8', errors='ignore')
    submitData.write("id,unknownEntities\n")
    testReader = csv.reader(testData)
    with torch.no_grad():
        for item in testReader:
            id, title, text = item[0], item[1], item[2]
            text = title + text
            batchSentence, lenList = dispose(text, config)
            batchSentence.to(DEVICE)
            tagScores  = net(batchSentence)
            
            sentenceArr, tagArr = []
            for index, element in enumerate(lenList):
                tagScore = tagScores[index][:element]
                sentence = batchSentence[index][:element]
                sentenceArr.append(sentence.cpu().numpy().tolist())
                tagArr.append([element.argmax().item() for element in tagScore])

            entityArr = acquireEntity(sentenceArr, tagArr, config)
            def filter_word(w):
                for wbad in ['？','《','🔺','️?','!','#','%','%','，','Ⅲ','》','丨','、','）','（','​',
                        '👍','。','😎','/','】','-','⚠️','：','✅','㊙️','“',')','(','！','🔥',',']:
                    if wbad in w:
                        return ''
                return w
            entityArr = [entity for entity in entityArr if filter_word(entity) != '']

            if len(entityArr) == 0: entityArr = ['FUCK']

            submitData.write('%s,%s\n' % (id, ';'.join(entityArr)))
    testData.close(); submitData.close();


if __name__ == "__main__":
    optParser = OptionParser()
    optParser.add_option('-tr','--train',action = 'store_true', dest='train')
    optParser.add_option('-te','--test',action = 'store_true', dest='test')

    f = open('./config.yml', encoding='utf-8', errors='ignore')
    config = yaml.load(f)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['DEVICE'] = DEVICE
    f.close()
    option , args = optParser.parse_args()

    if option.train == True:
        train(config)
        
    if option.test == True:
        test(config)

        

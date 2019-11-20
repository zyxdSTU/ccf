from data_loader import tagDict
from data_loader import tag2id
from data_loader import id2tag
import copy
import os
import torch
import torch.nn as nn
from torch.utils import data
from data_loader import testPad
from data_loader import pad
from data_loader import NERDataset
from data_loader import NERTestDataset
from itertools import chain
from bilstm import BiLSTM
from model_process import train
from model_process import test
from model_process import valid
from optparse import OptionParser
import yaml
from util import generateResult
from random import shuffle
from util import readData
from idcnn import IDCNN
import pandas
import numpy
from bilstm_attn import BiLSTM_ATTN

def run(dataDir, fold=5):
    f = open('./config.yml', encoding='utf-8', errors='ignore')
    config = yaml.safe_load(f)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['DEVICE'] = DEVICE
    batchSize = config['model']['batchSize']

    optParser = OptionParser()
    optParser.add_option('-m', '--model', action = 'store', type='string', dest ='modelName')
    option , args = optParser.parse_args()
    modelName = config['modelName'] = option.modelName

    #保存最终结果
    f = open(os.path.join(dataDir, modelName, 'result.txt'), 'w', encoding='utf-8', errors='ignore')

    #测试数据
    testDataPath = config['data']['testDataPath']
    testDataset = NERTestDataset(testDataPath, config)
    testIter = data.DataLoader(dataset = testDataset,
                                 batch_size = batchSize,
                                 shuffle = False,
                                 num_workers = 4,
                                 collate_fn = testPad)    

    for i in range(fold):
        print('--------------------第%d次验证-------------------\n' %(i+1))
        #验证数据
        validDataset = NERDataset(os.path.join(dataDir, str(i)+'.txt'), config)
        validIter = data.DataLoader(dataset = validDataset,
                                batch_size = batchSize,
                                shuffle = False,
                                num_workers = 4,
                                collate_fn = pad)
        #训练数据
        trainPathArr = [os.path.join(dataDir, str(j)+'.txt') for j in range(fold) if j != i]
        assert len(trainPathArr) == fold -1    
    
        trainDataset = NERDataset(trainPathArr, config)
        trainIter = data.DataLoader(dataset = trainDataset,
                                batch_size = batchSize,
                                shuffle = True,
                                num_workers = 4,
                                collate_fn = pad) 

        #加载网络
        if modelName == 'bilstm':
            net = BiLSTM(config)

        if modelName == 'idcnn':
            net = IDCNN(config)

        if modelName == 'bilstm_attn':
            net = BiLSTM_ATTN(config)

        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)

        net = net.to(DEVICE)

        config['submitPath'] = os.path.join(dataDir, modelName, str(i)+'.csv')
        config['modelSavePath'] = os.path.join(dataDir, modelName, str(i) + '.pkl')

        trainLoss, validLoss, f1Score, accurate, recall = train(net, trainIter, validIter, config)

        #验证集中实际效果
        modelSavePath = config['modelSavePath']
        if os.path.exists(modelSavePath):
            net.load_state_dict(torch.load(modelSavePath))

        #未过滤训练集实体的缺失比、过滤完训练集实体的缺失比
        disappear1, disappear2 = test(net,testIter, config)

        f.write('第%d次验证\n' % (i+1))
        f.write('trainLoss: %f\n' % trainLoss)
        f.write('validLoss: %f\n'% validLoss)
        f.write('f1Score %f, accurate %f, recall %f\n' %(f1Score, accurate, recall))
        f.write('测试集中缺失比%f %f\n' % (disappear1, disappear2))
        f.write('\n')

    f.close()

run('./data/5-fold')


    



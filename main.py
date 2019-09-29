import yaml
import os
from optparse import OptionParser
from data_loader import NERDataset
from data_loader import NERTestDataset
from data_loader import pad
from data_loader import testPad
from torch.utils import data
from tqdm import tgrange
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from data_loader import tagDict
import torch
import sys
import csv
from bilstm import BiLSTM
from bilstm import bilstm_train
from bilstm import bilstm_test
from bilstm_crf import bilstm_crf_train
from bilstm_crf import bilstm_crf_test
from bilstm_crf import BiLSTM_CRF
from random import random
from transformer_cnn import Transformer_CNN
from transformer_cnn import transformer_cnn_train
from transformer_cnn import transformer_cnn_test

def main(config):
    trainDataPath = config['data']['trainDataPath']
    validDataPath = config['data']['validDataPath']
    testDataPath = config['data']['testDataPath']
    batchSize = config['model']['batchSize']
    
    #GPU/CPU
    DEVICE = config['DEVICE']

    trianDataset = NERDataset(trainDataPath, config) 
    validDataset = NERDataset(validDataPath, config)
    testDataset = NERTestDataset(testDataPath, config)

    trainIter = data.DataLoader(dataset = trianDataset,
                                 batch_size = batchSize,
                                 shuffle = True,
                                 num_workers = 6,
                                 collate_fn = pad)

    validIter = data.DataLoader(dataset = validDataset,
                                 batch_size = batchSize,
                                 shuffle = False,
                                 num_workers = 6,
                                 collate_fn = pad)
    
    testIter = data.DataLoader(dataset = testDataset,
                                 batch_size = batchSize,
                                 shuffle = False,
                                 num_workers = 6,
                                 collate_fn = testPad)

    
    if config['modelName'] == 'bilstm':
        net = BiLSTM(config)
        config['modelSavePath'] = config['data']['BiLSTMSavePath']
        modelSavePath = config['modelSavePath']
        config['submitDataPath'] = config['data']['BiLSTMSubmitDataPath']
        train = bilstm_train
        test = bilstm_test

    if config['modelName'] == 'bilstm_crf':
        net = BiLSTM_CRF(config)
        config['modelSavePath'] = config['data']['BiLSTMCRFSavePath']
        modelSavePath = config['modelSavePath']
        config['submitDataPath'] = config['data']['BiLSTMCRFSubmitDataPath']
        train = bilstm_crf_train
        test = bilstm_crf_test

    if config['modelName'] == 'transformer_cnn':
        net = Transformer_CNN(config)
        config['modelSavePath'] = config['data']['TransformerCNNSavePath']
        config['submitDataPath'] = config['data']['TransformerCNNSubmitDataPath']
        modelSavePath = config['modelSavePath']
        train = transformer_cnn_train
        test = transformer_cnn_test


    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    net = net.to(DEVICE)

    if os.path.exists(modelSavePath):
        net.load_state_dict(torch.load(modelSavePath))

    #if config['train']: 
    #train(net, trainIter, validIter, config)

    #if config['test']: 
    test(net,testIter, config)

    
if __name__ == "__main__":
    optParser = OptionParser()
    optParser.add_option('--train',action = 'store_true', dest='train')
    optParser.add_option('--test',action = 'store_true', dest='test')
    optParser.add_option('-m', '--model', action = 'store', type='string', dest ='modelName')
    option , args = optParser.parse_args()
    f = open('./config.yml', encoding='utf-8', errors='ignore')
    config = yaml.safe_load(f)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['DEVICE'] = DEVICE
    config['modelName'] = option.modelName
    config['train'], config['test'] = option.train, option.test

    main(config)

    f.close()
    
        









        

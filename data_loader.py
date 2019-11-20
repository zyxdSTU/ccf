from torch.utils import data
from pytorch_pretrained_bert import BertTokenizer
import torch
import copy
from random import random
from transformers import XLNetTokenizer

tagDict = ['<PAD>', 'O', 'B', 'I', 'E']
#tagDict = ['<PAD>', 'B', 'I', 'E', 'S','C', 'J', 'F', 'T', 'O']

# tagDict = tagDict = ['<PAD>', 'B-NAME', 'M-NAME', 'E-NAME', 'O', 'B-CONT', 'M-CONT', 
#     'E-CONT', 'B-EDU', 'M-EDU', 'E-EDU', 'B-TITLE', 'M-TITLE', 'E-TITLE', 
#     'B-ORG', 'M-ORG', 'E-ORG', 'B-RACE', 'E-RACE', 'B-PRO', 'M-PRO', 'E-PRO', 
#     'B-LOC', 'M-LOC', 'E-LOC','S-RACE', 'S-NAME', 'M-RACE', 'S-ORG', 'S-CONT', 
#     'S-EDU','S-TITLE', 'S-PRO','S-LOC']

tag2id = {element:index for index, element in enumerate(tagDict)}
id2tag = {index:element for index, element in enumerate(tagDict)}


class NERDataset(data.Dataset):
    '''
    : path 语料路径
    '''
    def __init__(self, path, config):
        self.config = config
        if isinstance(path, list):
            data = ""
            for element in path:       
                f = open(element, 'r', encoding='utf-8', errors='ignore')
                data += f.read(); f.close()
            data = data.split('\n\n')
        else:
            f = open(path, 'r', encoding='utf-8', errors='ignore')
            data = f.read().split('\n\n'); f.close()

        data = [[(element2.split('\t')[0], element2.split('\t')[1])
            for element2 in element1.split('\n') if len(element2) != 0]
            for element1 in data if len(element1.strip('\n')) != 0 ]

        self.sentenceList = [[element2[0] for element2 in element1]for element1 in data]

        self.tagList = [[element2[1] for element2 in element1]for element1 in data]

        if self.config['model']['pretrained_model'] == 'Bert':
            self.tokenizer = BertTokenizer.from_pretrained(self.config['model']['bert_base_chinese'], do_lower_case=True)
        
        if self.config['model']['pretrained_model'] == 'XLNet':
            self.tokenizer = XLNetTokenizer.from_pretrained(self.config['model']['xlnet_base_chinese'], do_lower_case=False)
    
    def __len__(self):
        return len(self.sentenceList)

    def __getitem__(self, index):
        maxWordLen = self.config['model']['maxWordLen']
        sentence, tag = self.sentenceList[index], self.tagList[index]
        originSentence = copy.deepcopy(sentence)

        if self.config['model']['pretrained_model'] == 'Bert':
            #大写转小写
            sentence = list(''.join(sentence).lower())
            for i in range(len(sentence)):
                if sentence[i] in self.tokenizer.vocab.keys():
                    sentence[i] = self.tokenizer.vocab[sentence[i]]
                else: sentence[i] = self.tokenizer.vocab['[UNK]']

        if self.config['model']['pretrained_model'] == 'XLNet':
            sentence = self.tokenizer.encode(sentence)
            
        tag = [tag2id[element] for element in tag]

        #如果句子太长进行截断
        if len(sentence) > maxWordLen:
            sentence = sentence[:maxWordLen]
            tag = tag[:maxWordLen]

        assert len(sentence) == len(tag)

        return sentence, tag, originSentence


class NERTestDataset(data.Dataset):
    '''
    : path 语料路径
    '''
    def __init__(self, path, config):
        self.config = config
        f = open(path, 'r', encoding='utf-8', errors='ignore')
        data = f.read().split('\n\n'); f.close()

        self.sentenceList = [[element2 for element2 in element1.split('\n') if len(element2) != 0]for element1 in data if len(element1) != 0]

        if self.config['model']['pretrained_model'] == 'Bert':
            self.tokenizer = BertTokenizer.from_pretrained(self.config['model']['bert_base_chinese'], do_lower_case=True)
        
        if self.config['model']['pretrained_model'] == 'XLNet':
            self.tokenizer = XLNetTokenizer.from_pretrained(self.config['model']['xlnet_base_chinese'], do_lower_case=True)

    def __len__(self):
        return len(self.sentenceList)

    def __getitem__(self, index):
        maxWordLen = self.config['model']['maxWordLen']
        sentence = self.sentenceList[index]

        #如果句子太长进行截断
        if len(sentence) > maxWordLen:
            sentence = sentence[:maxWordLen]

        originSentence = copy.deepcopy(sentence)

        if self.config['model']['pretrained_model'] == 'Bert':
            #大写转小写
            sentence = list(''.join(sentence).lower())
          
            for i in range(len(sentence)):
                if sentence[i] in self.tokenizer.vocab.keys():
                    sentence[i] = self.tokenizer.vocab[sentence[i]]
                else: sentence[i] = self.tokenizer.vocab['[UNK]']

        if self.config['model']['pretrained_model'] == 'XLNet':
            sentence = self.tokenizer.encode(sentence)

        return sentence, originSentence

import yaml
f = open('./config.yml', encoding='utf-8', errors='ignore')
config = yaml.safe_load(f); f.close()

def testPad(batch):
    f1 = lambda x:[element[x] for element in batch]
    lenList = [len(element) for element in f1(0)]
    maxLen = max(lenList)

    if config['model']['pretrained_model'] == 'XLNet':
        f2 = lambda x, maxLen:[[0] * (maxLen - len(element[x])) + element[x] for element in batch]
    if config['model']['pretrained_model'] == 'Bert':
        f2 = lambda x, maxLen:[element[x] + [0] * (maxLen - len(element[x])) for element in batch]

    return torch.LongTensor(f2(0, maxLen)), f1(1), lenList
'''
进行填充
'''
def pad(batch):
    #句子最大长度
    f1 = lambda x:[element[x] for element in batch]
    lenList = [len(element) for element in f1(0)]

    
    maxLen = max(lenList)
    
    if config['model']['pretrained_model'] == 'XLNet':
        f2 = lambda x, maxLen:[[0] * (maxLen - len(element[x])) + element[x] for element in batch]
    if config['model']['pretrained_model'] == 'Bert':
        f2 = lambda x, maxLen:[element[x] + [0] * (maxLen - len(element[x])) for element in batch]


    return torch.LongTensor(f2(0, maxLen)), torch.LongTensor(f2(1, maxLen)), lenList, f1(2)



    


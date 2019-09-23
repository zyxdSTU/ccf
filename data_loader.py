from torch.utils import data
from pytorch_pretrained_bert import BertTokenizer
import torch
import copy
tagDict = ['<PAD>', 'O', 'B', 'I']

tag2id = {element:index for index, element in enumerate(tagDict)}
id2tag = {index:element for index, element in enumerate(tagDict)}


class NERDataset(data.Dataset):
    '''
    : path 语料路径
    '''
    def __init__(self, path, config):
        self.config = config
        f = open(path, 'r', encoding='utf-8', errors='ignore')
        sentenceList, tagList = [], []
        sentence, tag = [], []
        for line in f.readlines():
            #换行
            if len(line.strip()) == 0:
                if len(sentence) != 0 and len(tag) != 0: 
                    if len(sentence) == len(tag):
                        sentenceList.append(sentence); tagList.append(tag)
                sentence, tag = [], []
            else:
                line = line.strip()
                if len(line.split()) < 2: continue
                sentence.append(line.split('\t')[0])
                tag.append(line.split('\t')[1])
        f.close()
        if len(sentence) != 0 and len(tag) != 0: 
            if len(sentence) == len(tag):
                sentenceList.append(sentence); tagList.append(tag)

        self.tokenizer = BertTokenizer.from_pretrained(self.config['model']['bert_base_chinese'], do_lower_case=True)
        self.sentenceList, self.tagList = sentenceList, tagList
    
    def __len__(self):
        return len(self.sentenceList)

    def __getitem__(self, index):
        maxWordLen = self.config['model']['maxWordLen']
        sentence, tag = self.sentenceList[index], self.tagList[index]

        originSentence = copy.deepcopy(sentence)

        for i in range(len(sentence)):
            if sentence[i] in self.tokenizer.vocab.keys():
                sentence[i] = self.tokenizer.vocab[sentence[i]]
            else: sentence[i] = self.tokenizer.vocab['[UNK]']

        tag = [tag2id[element] for element in tag]

        assert len(sentence) == len(tag)
            
        #如果句子太长进行截断
        if len(sentence) > maxWordLen:
            sentence = sentence[:maxWordLen]
            tag = tag[:maxWordLen]
            
        return sentence, tag, originSentence


class NERTestDataset(data.Dataset):
    '''
    : path 语料路径
    '''
    def __init__(self, path, config):
        self.config = config
        f = open(path, 'r', encoding='utf-8', errors='ignore')
        data = f.read().split('\n\n'); f.close()

        self.sentenceList = [[element2 for element2 in element1.split('\n')]for element1 in data]
        self.tokenizer = BertTokenizer.from_pretrained(self.config['model']['bert_base_chinese'], do_lower_case=True)
    
    def __len__(self):
        return len(self.sentenceList)

    def __getitem__(self, index):
        maxWordLen = self.config['model']['maxWordLen']
        sentence = self.sentenceList[index]

        #如果句子太长进行截断
        if len(sentence) > maxWordLen:
            sentence = sentence[:maxWordLen]

        originSentence = copy.deepcopy(sentence)

        for i in range(len(sentence)):
            if sentence[i] in self.tokenizer.vocab.keys():
                sentence[i] = self.tokenizer.vocab[sentence[i]]
            else: sentence[i] = self.tokenizer.vocab['[UNK]']

        return sentence, originSentence

def testPad(batch):
    f1 = lambda x:[element[x] for element in batch]
    lenList = [len(element) for element in f1(0)]
    maxLen = max(lenList)

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
    
    f2 = lambda x, maxLen:[element[x] + [0] * (maxLen - len(element[x])) for element in batch]

    return torch.LongTensor(f2(0, maxLen)), torch.LongTensor(f2(1, maxLen)), lenList, f1(2)



    


from torch.utils import data
from pytorch_pretrained_bert import BertTokenizer
import torch

tagDict = ['<PAD>', 'O', 'S-C', 'B-C', 'I-C', 'E-C', 'S-TN', 'B-TN', 'I-TN', 'E-TN', 'S-TC', 
    'B-TC', 'I-TC', 'E-TC', 'S-TV', 'B-TV', 'I-TV', 'E-TV', 'S-VTN', 'B-VTN', 'I-VTN', 'E-VTN', 
    'S-VTC', 'B-VTC', 'I-VTC','E-VTC', 'S-VTV', 'B-VTV', 'I-VTV', 'E-VTV', 'S-VTL', 'B-VTL', 
    'I-VTL', 'E-VTL', 'S-VTS', 'B-VTS', 'I-VTS', 'E-VTS', 'S-VTE', 'B-VTE', 'I-VTE', 'E-VTE', 
    'S-S', 'B-S', 'I-S', 'E-S', 'S-TS', 'B-TS', 'I-TS', 'E-TS', 'S-TL', 'B-TL', 'I-TL', 'E-TL',
    'S-TE', 'B-TE', 'I-TE', 'E-TE', 'S-N', 'B-N', 'I-N','E-N', 'S-NC', 'B-NC', 'I-NC', 'E-NC',
    'S-NP', 'B-NP', 'I-NP', 'E-NP', 'S-NV', 'B-NV', 'I-NV', 'E-NV','S-NH', 'B-NH', 'I-NH', 
    'E-NH', 'S-L', 'B-L', 'I-L', 'E-L', 'S-V', 'B-V', 'I-V', 'E-V', 'S-DE', 'B-DE', 'I-DE',
    'E-DE', 'S-DE1', 'B-DE1', 'I-DE1', 'E-DE1', 'S-DE2', 'B-DE2', 'I-DE2', 'E-DE2', 
    'S-D', 'B-D', 'I-D', 'E-D', 'S-R', 'B-R', 'I-R', 'E-R']

tagDict = ['<PAD>', 'B-NAME', 'M-NAME', 'E-NAME', 'O', 'B-CONT', 'M-CONT', 
    'E-CONT', 'B-EDU', 'M-EDU', 'E-EDU', 'B-TITLE', 'M-TITLE', 'E-TITLE', 
    'B-ORG', 'M-ORG', 'E-ORG', 'B-RACE', 'E-RACE', 'B-PRO', 'M-PRO', 'E-PRO', 
    'B-LOC', 'M-LOC', 'E-LOC','S-RACE', 'S-NAME', 'M-RACE', 'S-ORG', 'S-CONT', 
    'S-EDU','S-TITLE', 'S-PRO','S-LOC']

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
                sentence.append(line.split()[0])
                tag.append(line.split()[1])
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
        tokens = self.tokenizer.tokenize(' '.join(sentence))
        sentence = self.tokenizer.convert_tokens_to_ids(tokens)
        tag = [tag2id[element] for element in tag]
        #如果句子太长进行截断
        if len(sentence) > maxWordLen:
            sentence = sentence[:maxWordLen]
            tag = tag[:maxWordLen]
        return sentence, tag

'''
进行填充
'''
def pad(batch):
    #句子最大长度
    f = lambda x:[element[x] for element in batch]
    lenList = [len(element) for element in f(0)]
    maxLen = max(lenList)
    
    f = lambda x, maxLen:[element[x] + [0] * (maxLen - len(element[x])) for element in batch]

    return torch.LongTensor(f(0, maxLen)), torch.LongTensor(f(1, maxLen)), lenList
import csv
import random
import re
import copy
from pytorch_pretrained_bert import BertTokenizer
import torch
import copy
from data_loader import id2tag

def cutData(originPath, trainPath, validPath, scale=0.9):

    origin = open(originPath, 'r', encoding='utf-8', errors='ignore')
    train = open(trainPath, 'w', encoding='utf-8', errors='ignore')
    valid = open(validPath, 'w', encoding='utf-8', errors='ignore')

    originReader = csv.reader(origin)
    trainWriter = csv.writer(train)
    validWriter = csv.writer(valid)
    
    #去除首行
    total = list(originReader)[1:]

    order = list(range(len(total)))
    random.shuffle(order)

    trainData = [total[order[index]] for index in order[:int(scale*len(total))]]
    validData = [total[order[index]] for index in order[int(scale*len(total)):]]
    
    for element in trainData:
        trainWriter.writerow(element)
    
    for element in validData:
        validWriter.writerow(element)

    origin.close(); train.close(); valid.close()

#cutData('./data/Train_Data.csv', './data/train.csv', './data/valid.csv')

def stop_words(x):
    try:
        x = x.strip()
    except:
        return ''
    #去除空格
    x = re.sub(r'\?+','',x)

    x = re.sub(r'\{IMG:.*?\}','',x)

    x = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', x)

    x = re.sub("[\w!#$%&'*+/=?^_`{|}~-]+(?:\.[\w!#$%&'*+/=?^_`{|}~-]+)*@(?:[\w](?:[\w-]*[\w])?\.)+[\w](?:[\w-]*[\w])?"
            ,'', x)    

    x = re.sub("0\d{2}-\d{8}|0\d{3}-\d{7}|\d{5}-\d{5}", '', x) 

    x = re.sub("(20\d{2}([\.\-/|年月\s]{1,3}\d{1,2}){2}日?(\s?\d{2}:\d{2}(:\d{2})?)?)|(\d{1,2}\s?(分钟|小时|天)前)"
            ,'', x) 

    return x


def acquireEntity(sentenceArr, tagArr, config):
    entityArr, entity = [], ''
    #print (tagArr)
    for i in range(len(tagArr)):
        for j in range(len(tagArr[i])):
            if tagArr[i][j] == 'B':
                if entity != '':entityArr.append(entity); entity = sentenceArr[i][j]
                else: entity += sentenceArr[i][j]
            if tagArr[i][j] == 'I':
                if entity != '': entity = entity + sentenceArr[i][j];

    if entity != '': entityArr.append(entity)

    return list(set(entityArr))

    
#生成测试数据
def dispose(x, config):
    sentenceArr = []

    #数据清洗
    x = stop_words(x)

    #切分句子
    pattern = r';|\.|\?|!|；|。|？|！'
    sentenceArr.extend([list(element) for element in re.split(pattern, x) if len(element.strip()) != ''])

    originSentenceArr = copy.deepcopy(sentenceArr)

    #字符转为相应的标识
    tokenizer = BertTokenizer.from_pretrained(config['model']['bert_base_chinese'], do_lower_case=True)
    for i in range(len(sentenceArr)):
        for j in range(len(sentenceArr[i])):
            if sentenceArr[i][j] in tokenizer.vocab.keys():
                sentenceArr[i][j] = tokenizer.vocab[sentenceArr[i][j]]
            else: sentenceArr[i][j] = tokenizer.vocab['[UNK]']
    
    #截断和补全
    maxWordLen = config['model']['maxWordLen']
    lenList = [len(element)for element in sentenceArr]
    maxLen = maxWordLen if max(lenList) > maxWordLen else max(lenList)
    lenList = [element if element < maxLen else maxLen for element in lenList]
    sentenceArr = [element+[0]*(maxLen-len(element)) if len(element) < maxLen else element[:maxLen] for element in sentenceArr]


    return torch.LongTensor(sentenceArr), originSentenceArr, lenList


def dataPrepare(inputPath, outputPath):
    input = open(inputPath, 'r', encoding='utf-8', errors='ignore')
    output = open(outputPath, 'w', encoding='utf-8', errors='ignore')
    inputReader = csv.reader(input)
    pattern = r';|\.|\?|!|；|。|？|！'
    maxLen = 0
    for item in inputReader:
        id, title, text = item[0], item[1], item[2]
        sentenceArr, tagArr = [], []

        #去除一些冗余信息
        title = stop_words(title); text = stop_words(text)

        #注意过滤空行
        if len(title) != 0: sentenceArr.extend([element for element in re.split(pattern, title) if len(element.strip()) != ''])
        if len(text) != 0: sentenceArr.extend([element for element in re.split(pattern, text) if len(element.strip()) != '']);

        # Len = max([len(element) for element in sentenceArr])
        # if Len > maxLen: maxLen = Len

        if len(item[3].strip()) == 0: 
            tagArr = [['O'] * len(sentence) for sentence in sentenceArr]
        else:
            entityArr = item[3].split(';')
            tagArr = [sentence for sentence in sentenceArr]
            for entity in entityArr:
                for i in  range(len(tagArr)):
                    tagArr[i] = tagArr[i].replace(entity, 'Ё' + (len(entity)-1)*'Ж')

            for i in range(len(tagArr)):
                tagArr[i] = list(tagArr[i])
                for j in range(len(tagArr[i])):
                    if tagArr[i][j] == 'Ё': tagArr[i][j] = 'B'
                    elif tagArr[i][j] == 'Ж': tagArr[i][j] = 'I'
                    else: tagArr[i][j] = 'O'
                        
        for sentence, tag in zip(sentenceArr, tagArr):
            assert len(sentence) == len(tag)
            for element1, element2 in zip(sentence, tag):
                output.write(element1 + '\t' + element2 + '\n')
            output.write('\n')
    #print(maxLen)
    input.close(); output.close()
              
dataPrepare('./data/train.csv', './data/train.txt')
dataPrepare('./data/valid.csv', './data/valid.txt')

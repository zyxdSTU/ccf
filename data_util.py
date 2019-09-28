import csv
from random import random
from random import shuffle
import re
import copy
from pytorch_pretrained_bert import BertTokenizer
import torch
import copy
from data_loader import id2tag
import pandas as pd

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
    
    #html标识
    x = re.sub(r'<.*?>', '', x)
    
    x = re.sub(r'&ensp;|&emsp;|&nbsp;|&lt;|&gt;|&amp;|&quot;|&copy;|&reg;', '',x)
    
    #Imag标识和代码
    x = re.sub(r'\{.*?\}', '', x)
    
    #网址
    x = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', x)
    
    #去除空格
    x = x.replace(' ', '')

    return x


def acquireEntity(sentenceArr, tagArr, method='BIOES'):
    entityArr, entity = [], ''
    for i in range(len(tagArr)):
        for j in range(len(tagArr[i])):
            if method == 'BIO':
                if tagArr[i][j] == 'B':
                    if entity != '':entityArr.append(entity.strip()); entity = sentenceArr[i][j]
                    else: entity += sentenceArr[i][j]

                if tagArr[i][j] == 'I':
                    if entity != '': entity = entity + sentenceArr[i][j]

                if tagArr[i][j] == 'O':
                    if entity != '': entityArr.append(entity.strip()); entity = ''

                if entity != '': entityArr.append(entity.strip())

            elif method == 'BIOES':
                if tagArr[i][j] == 'S': 
                    entity = ''; entityArr.append(sentenceArr[i][j])
                if tagArr[i][j] == 'B': 
                    entity = sentenceArr[i][j]
                if tagArr[i][j] == 'I':
                    if entity != '': entity += sentenceArr[i][j]
                if tagArr[i][j] == 'E': 
                    if entity != '': entity += sentenceArr[i][j]; entityArr.append(entity); entity = ''
                if tagArr[i][j] == 'O': 
                    if entity != '': entity = ''
    entityArr = [entity.strip() for entity in entityArr]
    return list(set(entityArr))

#剔除重复的实体
def f2_score(y_true, y_pred, y_Sentence, validLenPath):
    validLen = open(validLenPath, 'r', encoding='utf-8', errors='ignore')
    start = 0
    TP, FP, FN = 0, 0, 0
    for line in validLen.readlines():
        id, length = line.strip('\n').split('\t')[0], int(line.strip('\n').split('\t')[1])
        trueEntity = acquireEntity(y_Sentence[start:start+length], y_true[start:start+length])
        predEntity = acquireEntity(y_Sentence[start:start+length], y_pred[start:start+length])
        start = start + length

        TPE, FPE, FNE = 0, 0, 0
        for entity in predEntity:
            if entity in trueEntity: TPE += 1
            else: FPE += 1
        FNE = len(trueEntity) - TPE
        TP += TPE; FP += FPE; FN += FNE
    
    if TP + FP == 0 or TP + FN == 0: return 0

    MicroP = TP / (TP + FP); MicroR = TP / (TP + FN)

    if MicroP + MicroR == 0: return 0

    MicroF = 2 * MicroP * MicroR / (MicroP + MicroR)

    validLen.close()

    return MicroF

##Test数据集需要记住每一项数据包含的文本行数
##Test数据集并且没有标签
def dataTestPrepare(TrainDataPath, inputPath, outputDataPath, outputLenPath):

    input = open(inputPath, 'r', encoding='utf-8', errors='ignore')
    outputData = open(outputDataPath, 'w', encoding='utf-8', errors='ignore')
    outputLen = open(outputLenPath, 'w', encoding='utf-8', errors='ignore')


    inputReader = csv.reader(input)
    pattern = r';|\?|!|；|。|？|！'

    for item in inputReader:
        if inputReader.line_num == 1: continue
        id, title, text = item[0], item[1], item[2]
        sentenceArr, tagArr = [], []

        if len(title) == 0: string = text
        elif len(text) == 0: string = title
        else: string = title +'。'+ text

        string = stop_words(string)
        sentenceArr = re.split(pattern, string)

        #过滤超长句子
        sentenceArr = [element.strip() for element in sentenceArr]
        sentenceArr = [element for element in sentenceArr if len(element) > 0 and len(element) <= 200]
        
        for sentence in sentenceArr:
            for element in sentence:
                outputData.write(element + '\n')
            outputData.write('\n')

        outputLen.write(id + '\t' + str(len(sentenceArr)) + '\n')
        
    input.close(); outputData.close(); outputLen.close()


def dataPrepare(inputPath, trainPath, validPath,  method ='BIOES', portion = 0.9):

    def contain(sentence, entityArr):
        for entity in entityArr:
            if entity in sentence: return True
        return False

    input = open(inputPath, 'r', encoding='utf-8', errors='ignore')
    trainOutput = open(trainPath, 'w', encoding='utf-8', errors='ignore')
    validOutput = open(validPath, 'w', encoding='utf-8', errors='ignore')

    inputReader = csv.reader(input)
    pattern = r';|\?|!|；|。|？|！'

    sentenceArrTotal, tagArrTotal = [], []
    for item in inputReader:
        if inputReader.line_num == 1: continue
        id, title, text = item[0], item[1], item[2]
        sentenceArr, tagArr = [], []

        if len(title) == 0: string = text
        elif len(text) == 0: string = title
        else: string = title +'。'+ text

        string = stop_words(string)
        sentenceArr = re.split(pattern, string)

        #处理句子、过滤超长句子
        sentenceArr = [element for element in sentenceArr if len(element) > 0 and len(element) <= 200]

        if len(item[3].strip()) == 0: tagArr = [['O'] * len(sentence) for sentence in sentenceArr]            
        else:
            #实体按长度排序
            entityArr = item[3].split(';')
            entityArr = sorted(entityArr,key = lambda i:len(i),reverse=True) 

            tagArr = [sentence for sentence in sentenceArr]
            for entity in entityArr:
                for i in  range(len(tagArr)):
                    if method == 'BIO':
                        tagArr[i] = tagArr[i].replace(entity, 'Ё' + (len(entity)-1)*'Ж')
                    elif method == 'BIOES':
                        tagArr[i] = tagArr[i].replace(entity, 'Ё' + (len(entity)-2) * 'Ж' + 'З') if len(entity) > 1 else tagArr[i].replace(entity, 'И')

            for i in range(len(tagArr)):
                tagArr[i] = list(tagArr[i])
                for j in range(len(tagArr[i])):
                    if tagArr[i][j] == 'Ё': tagArr[i][j] = 'B'
                    elif tagArr[i][j] == 'Ж': tagArr[i][j] = 'I'
                    elif tagArr[i][j] == 'З': tagArr[i][j] = 'E'
                    elif tagArr[i][j] == 'И': tagArr[i][j] = 'S'
                    else: tagArr[i][j] = 'O'

        assert len(sentenceArr) == len(tagArr)
        
        sentenceArrTotal.extend(sentenceArr); tagArrTotal.extend(tagArr)

    #随机去除不包含实体的句子
    sentenceArrTemp, tagArrTemp = [], []
    for sentence, tag in zip(sentenceArrTotal, tagArrTotal):
        assert len(sentence) == len(tag)
        if ''.join(list(set(tag))) == 'O' and random() < 0.5: continue
        sentenceArrTemp.append(sentence); tagArrTemp.append(tag)
    sentenceArrTotal, tagArrTotal = sentenceArrTemp, tagArrTemp

    data = [(sentence, tag) for sentence, tag in zip(sentenceArrTotal, tagArrTotal)]
    #print (len(data))
    shuffle(data)

    trainData = data[:int(len(data) * portion)]
    validData = data[int(len(data)* portion):]

    for element in trainData:
        for word, tag in zip(element[0], element[1]):
            trainOutput.write(word + '\t' + tag + '\n')
        trainOutput.write('\n')

    for element in validData:
        for word, tag in zip(element[0], element[1]):
            validOutput.write(word + '\t' + tag + '\n')
        validOutput.write('\n')

    input.close(); trainOutput.close(); validOutput.close()
    
              
dataPrepare('./data/Train_Data.csv', './data/train_random.txt', './data/valid_random.txt',method='BIOES', portion=0.9)

#dataPrepare('./data/Train_Data.csv', './data/valid.csv', './data/valid_bioes_fix.txt', './data/valid.record', method='BIOES')
#dataTestPrepare('./data/Test_Data.csv', './data/test.txt', './data/test.record')

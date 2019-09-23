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


def acquireEntity(sentenceArr, tagArr):
    entityArr, entity = [], ''
    for i in range(len(tagArr)):
        for j in range(len(tagArr[i])):

            if tagArr[i][j] == 'B':
                if entity != '':entityArr.append(entity.strip()); entity = sentenceArr[i][j]
                else: entity += sentenceArr[i][j]

            if tagArr[i][j] == 'I':
                if entity != '': entity = entity + sentenceArr[i][j]

            if tagArr[i][j] == 'O':
                if entity != '': entityArr.append(entity.strip()); entity = ''

    if entity != '': entityArr.append(entity.strip())

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
def dataTestPrepare(inputPath, outputDataPath, outputLenPath):

    input = open(inputPath, 'r', encoding='utf-8', errors='ignore')
    outputData = open(outputDataPath, 'w', encoding='utf-8', errors='ignore')
    outputLen = open(outputLenPath, 'w', encoding='utf-8', errors='ignore')


    inputReader = csv.reader(input)
    pattern = r';|\.|\?|!|；|。|？|！'

    for item in inputReader:
        if inputReader.line_num == 1: continue
        id, title, text = item[0], item[1], item[2]
        sentenceArr, tagArr = [], []

        #去除一些冗余信息
        title = stop_words(title); text = stop_words(text)

        #注意过滤空行
        if len(title) != 0: sentenceArr.extend([element for element in re.split(pattern, title) 
            if len(element.strip()) > 0])
        if len(text) != 0: sentenceArr.extend([element for element in re.split(pattern, text) 
            if len(element.strip()) > 0])

        
        for sentence in sentenceArr:
            for element in sentence:
                outputData.write(element + '\n')
            outputData.write('\n')

        outputLen.write(id + '\t' + str(len(sentenceArr)) + '\n')
        
    input.close(); outputData.close(); outputLen.close()


def dataPrepare(inputPath, outputPath, outputLenPath):

    def contain(sentence, entityArr):
        for entity in entityArr:
            if entity in sentence: return True
        return False

    input = open(inputPath, 'r', encoding='utf-8', errors='ignore')
    output = open(outputPath, 'w', encoding='utf-8', errors='ignore')

    outputLen = open(outputLenPath, 'w', encoding='utf-8', errors='ignore')

    inputReader = csv.reader(input)
    pattern = r';|\.|\?|!|；|。|？|！'

    #sentenceLen = open('./data/len.txt', 'a', encoding='utf-8', errors='ignore')
    for item in inputReader:
        id, title, text = item[0], item[1], item[2]
        sentenceArr, tagArr = [], []

        #去除一些冗余信息
        title = stop_words(title); text = stop_words(text)

        #注意过滤不足10字符的句子
        if len(title) != 0: sentenceArr.extend([element for element in re.split(pattern, title) 
            if (len(element.strip()) > 10)])
        if len(text) != 0: sentenceArr.extend([element for element in re.split(pattern, text) 
            if (len(element.strip()) > 10)])

        #for sentence in sentenceArr: sentenceLen.write(str(len(sentence)) + '\n')

        #不过滤不包含实体的句子
        if len(item[3].strip()) == 0: tagArr = [['O'] * len(sentence) for sentence in sentenceArr]            
        else:
            entityArr = item[3].split(';')

            #过滤不包含实体的句子
            #sentenceArr = [sentence for sentence in sentenceArr if contain(sentence, entityArr)]
            
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

        assert len(sentenceArr) == len(tagArr)

        outputLen.write(id + '\t' + str(len(sentenceArr)) + '\n')

        for sentence, tag in zip(sentenceArr, tagArr):
            assert len(sentence) == len(tag)
            for element1, element2 in zip(sentence, tag):
                output.write(element1 + '\t' + element2 + '\n')
            output.write('\n')

    input.close(); output.close(); outputLen.close()
    #sentenceLen.close()
              
dataPrepare('./data/train.csv', './data/train.txt', './data/train.record')
dataPrepare('./data/valid.csv', './data/valid.txt', './data/valid.record')
dataTestPrepare('./data/Test_Data.csv', './data/test.txt', './data/test.record')
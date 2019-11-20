import csv
from random import random
from random import shuffle
import re
import copy
from pytorch_pretrained_bert import BertTokenizer
import torch
from zhon.hanzi import punctuation
import pandas as pd

#找到一些非法字符
def findIllegalWord(trainDataPath, testDataPath):
    import string
    additional_chars = set()
    trainDF = pd.read_csv(trainDataPath)
    testDF = pd.read_csv(testDataPath)
    trainDF['text'] =  trainDF['title'].fillna('') + trainDF['text'].fillna('')
    testDF['text'] =  testDF['title'].fillna('') + testDF['text'].fillna('')
    for t in list(trainDF['text']) + list(testDF['text']):
        additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', t))
    extra_chars = set(punctuation + string.punctuation)
    additional_chars = additional_chars.difference(extra_chars)
    return additional_chars

ilegalWordSet = findIllegalWord('./data/Train_Data.csv', './data/Test_Data.csv')

#去除一些停用词
def stop_words(x, illegalWordSet):
    import string
    try:
        x = x.strip()
    except:
        return ''
    #去除?????
    x = re.sub(r'\?+','',x)
    
    #html标识
    x = re.sub(r'<.*?>', '', x)
    
    x = re.sub(r'&ensp;|&emsp;|&nbsp;|&lt;|&gt;|&amp;|&quot;|&copy;|&reg;', '',x)
    
    #Imag标识和代码
    x = re.sub(r'\{.*?\}', '', x)
    
    #网址
    x = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', x)
    
    #去除特殊字符
    x = re.sub(r"\t|\n|\b|\xa0|\x0b|\x1c|\x1d|\x1e|\xe2\x80\x8b|\xe2\x80\x8c|\xe2\x80\x8d|\u200B|‼️​", "", x)

    #去除空格
    x = x.replace(' ', '')

    for word in illegalWordSet:
        x.replace(word, '')

    return x


#拆分一行句子
def disposeLine(string, minLen = 5, maxLen=200):
    #去除停用词等
    string = stop_words(string, illegalWordSet=ilegalWordSet)

    #分句
    pattern = r';|\?|!|；|。|？|！'
    sentenceArr = re.split(pattern, string)

    #合并短句子
    temp, tempArr = "", []
    for element in sentenceArr:
        if len(temp) + len(element) <= maxLen:
            temp = element if len(temp)==0 else temp+ '。' + element
        else: 
            tempArr.append(temp)
            if len(element) > maxLen: tempArr.append(element); temp = ""
            else: temp = element
    if len(temp) != 0: tempArr.append(temp)
    sentenceArr = tempArr

    #处理短句子和超长句子
    sentenceArr = [element.strip() for element in sentenceArr if len(element.strip()) >= minLen]
    sentenceArr = [element if len(element) <= maxLen else element[:maxLen] for element in sentenceArr]

    return sentenceArr

#使用BIOES标注方法标注测试集，outputLen用于生成提交文件，说明参考util.py中generateResult函数
def dataTestPrepare(inputDataPath, outputDataPath, outputLenPath):

    input = open(inputDataPath, 'r', encoding='utf-8', errors='ignore')
    output = open(outputDataPath, 'w', encoding='utf-8', errors='ignore')
    outputLen = open(outputLenPath, 'w', encoding='utf-8', errors='ignore')

    inputReader = csv.reader(input)

    for item in inputReader:
        if inputReader.line_num == 1: continue
        id, title, text = item[0], item[1], item[2]
        sentenceArr, tagArr = [], []

        if len(title) == 0: string = text
        elif len(text) == 0: string = title
        else: string = title +'。'+ text

        sentenceArr = disposeLine(string)
        
        for sentence in sentenceArr:
            for element in sentence:
                output.write(element + '\n')
            output.write('\n')

        outputLen.write(id + '\t' + str(len(sentenceArr)) + '\n')
        
    input.close(); output.close(); outputLen.close()

#使用BIOES标注方法标注训练集
def dataPrepare(inputDataPath, outputDataPath, outputLenPath, method ='BIOES'):
    input = open(inputDataPath, 'r', encoding='utf-8', errors='ignore')
    output = open(outputDataPath, 'w', encoding='utf-8', errors='ignore')
    outputLen = open(outputLenPath, 'w', encoding='utf-8', errors='ignore')

    inputReader = csv.reader(input)

    sentenceArrTotal, tagArrTotal = [], []
    for item in inputReader:
        #过滤第一行列名
        if inputReader.line_num == 1: continue
        id, title, text = item[0], item[1], item[2]

        #实体按长度排序，并且过滤掉不包含实体的列
        if item[3] != '':
            entityArr = item[3].split(';')
            entityArr = [element for element in entityArr if len(element) > 1]
            entityArr = sorted(entityArr,key = lambda i:len(i),reverse=True)
        else: continue 

        sentenceArr, tagArr = [], []

        if len(title) == 0: string = text
        elif len(text) == 0: string = title
        else: string = title +'。'+ text

        #切分文本为多个句子
        sentenceArr = disposeLine(string)

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
        
        for sentence, tag in zip(sentenceArr, tagArr):
            for element1, element2 in zip(sentence, tag):
                output.write(element1 + '\t' + element2 + '\n')
            output.write('\n')

        #加上真实实体
        outputLen.write(id + '\t' + str(len(sentenceArr)) + '\t' + item[3] +'\n')

    input.close(); output.close(); outputLen.close()


import os

def cutData(dataPath, saveDir, fold=5):
    totalPath = os.path.join(saveDir, 'total.txt')
    totalLenPath = os.path.join(saveDir, 'total.len')
    dataPrepare(dataPath, totalPath, totalLenPath)
    f = open(totalPath, 'r', encoding='utf-8', errors='ignore')
    data = f.read().split('\n\n'); f.close()
    data = [[(element2.split('\t')[0], element2.split('\t')[1])
            for element2 in element1.split('\n') if len(element2) != 0]
            for element1 in data if len(element1.strip('\n')) != 0 ]

    sentenceList = [[element2[0] for element2 in element1]for element1 in data]
    tagList = [[element2[1] for element2 in element1]for element1 in data]

    data = [(sentence, tag) for sentence, tag in zip(sentenceList, tagList)]

    #shuffle数据
    shuffle(data)

    length = len(data) #文件行数
    perLength = length // fold if length % fold == 0 else  length // fold + 1
    start, end = 0, 0
    for i in range(fold):
        end = start + perLength
        end = end if end <= length else length
        path = os.path.join(saveDir, '%d.txt' % i)
        f = open(path, 'w', encoding='utf-8', errors='ignore')
        for sentence, tag in data[start: end]:
            for element1, element2 in zip(sentence, tag):
                f.write(element1 + '\t' + element2 + '\n')
            f.write('\n')
        f.close()
        start = end 

# dataTestPrepare('./data/Test_Data.csv', './data/test.txt', './data/test.len')
# cutData('./data/Train_Data.csv', './data/5-fold')

from zhon.hanzi import punctuation
import pandas as pd
from data_loader import id2tag
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from tqdm import tqdm
from random import shuffle
from sklearn.utils import shuffle
def readData(path):
    f = open(path, 'r', encoding='utf-8', errors='ignore')
    data = f.read().split('\n\n'); f.close()

    data = [[(element2.split('\t')[0], element2.split('\t')[1])
        for element2 in element1.split('\n') if len(element2) != 0]
        for element1 in data if len(element1.strip('\n')) != 0 ]

    sentenceList = [[element2[0] for element2 in element1]for element1 in data]

    tagList = [[element2[1] for element2 in element1]for element1 in data]

    return sentenceList, tagList
    
#找出所有旧实体
def oldEntities(dataPath):
    #找出所有旧实体
    old_entities = []
    train_df = pd.read_csv(dataPath, header=0)
    for x in list(train_df["unknownEntities"].fillna("")):
        old_entities.extend(x.split(";"))
    old_entities = [element for element in old_entities if len(element.strip()) != 0]
    return old_entities
old_entities = oldEntities('./data/Train_Data.csv')

#形成最后提交结果
def generateResult(sentenceArr, tagArr,  config):
    def filter_word(w):
        import string
        for word in w:
            if word in ['？','《','🔺','!','#','%','，','Ⅲ','》','丨','、','​', '…',
                    '👍','。','😎','/','】','-','⚠️','：','✅','㊙️','！','🔥',',',
                    '.','——', '“', '”', '！', ' ']:
                return ''
            if word in ['(', ')', '（', '）', '?']: continue
            if word in punctuation + string.punctuation: return ''
        return w
    
    testLenPath = config['data']['testLenPath']
    submitData = open(config['submitPath'], 'w', encoding='utf-8', errors='ignore')
    testLen = open(testLenPath, 'r', encoding='utf-8', errors='ignore')
    submitData.write('id,unknownEntities\n')

    lenList = []
    start, end = 0, 0
    total, count1, count2 = 0, 0, 0
    for line in testLen.readlines():
        id, length = line.strip('\n').split('\t')[0], int(line.strip('\n').split('\t')[1])
        sentenceElement, tagElement = sentenceArr[start:start+length], tagArr[start:start+length]
        start += length
        entityArr = acquireEntity(sentenceElement, tagElement)

        #过滤无用实体
        entityArr = [entity for entity in entityArr if filter_word(entity) != '']

        if len(entityArr) == 0: count1 += 1;

        #过滤旧实体
        entityArrTemp = [entity for entity in entityArr if entity not in old_entities]

        total += 1
        if len(entityArrTemp) == 0: count2 += 1;

        submitData.write('%s,%s\n' % (id, ';'.join(entityArr)))
    submitData.close(); testLen.close()
    
    print (count1, count2, total)
    print('缺失比1 :%f\n' % (count1*1.0/total * 100))
    print('缺失比2 :%f\n' % (count2*1.0/total * 100))
    print (count1, count2, total)
    return round(count1 * 1.0 /total * 100, 2), round(count2 * 1.0 /total * 100, 2)

#获取实体
def acquireEntity(sentenceArr, tagArr, method='BIOES'):
    entityArr, entity = [], ''
    for i in range(len(tagArr)):
        for j in range(len(tagArr[i])):
            if method == 'BIO':
                if tagArr[i][j] == 'B':
                    if entity != '':entityArr.append(entity); entity = sentenceArr[i][j]
                    else: entity += sentenceArr[i][j]

                if tagArr[i][j] == 'I':
                    if entity != '': entity = entity + sentenceArr[i][j]

                if tagArr[i][j] == 'O':
                    if entity != '': entityArr.append(entity); entity = ''

            elif method == 'BIOES':
                if tagArr[i][j] == 'S': 
                    entity = ''; entityArr.append(sentenceArr[i][j])
                if tagArr[i][j] == 'B': 
                    entity = sentenceArr[i][j]
                if tagArr[i][j] == 'I':
                    if entity != '': entity += sentenceArr[i][j]
                if tagArr[i][j] == 'E': 
                    if entity != '': entity += sentenceArr[i][j]; entityArr.append(entity); entity = ''
                if tagArr[i][j] not in ['B', 'E', 'I', 'S']: 
                    if entity != '': entity = ''

        if method == 'BIO' and entity != '': entityArr.append(entity)
        
    entityArr = [entity.strip() for entity in entityArr if len(entity.strip()) > 1]
    return list(set(entityArr))

#针对BIOES标注,计算F1值
def f1_score(y_true, y_pred):
    def entitys(seq):
        entityArr, entity = [], []
        for index, element in enumerate(seq):
            index = str(index)
            if element == 'S': 
                entity = []; entityArr.append(index)
            if element == 'B': 
                entity = [index]
            if element == 'I':
                if len(entity) != 0: entity.append(index)
            if element == 'E': 
                if len(entity) != 0: entity.append(index); entityArr.append('@'.join(entity)); entity = []
            if element not in ['B', 'E', 'I', 'S']: 
                if len(entity) != 0: entity = []
        return entityArr
    TP, FP, FN = 0, 0, 0

    for trueElement, preElement in zip(y_true, y_pred):
        trueEntitys = entitys(trueElement)
        preEntitys = entitys(preElement)
        TPE = 0
        for element in preEntitys:
            if element in trueEntitys: TPE += 1
        FPE = len(preEntitys) - TPE
        FNE = len(trueEntitys) - TPE

        TP += TPE; FP += FPE; FN += FNE

    if TP+FP == 0 or TP+FN == 0: return 0, 0, 0

    MicroP = TP / (TP + FP); MicroR = TP / (TP + FN)

    
    if MicroP + MicroR == 0: return 0, 0, 0

    MicroF = 2 * MicroP * MicroR / (MicroP + MicroR)

    return MicroF, MicroP, MicroR


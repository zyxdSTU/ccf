import codecs
import os
from seqeval.metrics import f1_score, accuracy_score, classification_report
import sys
sys.path.append('../')
from data_util import acquireEntity

#3 crf train
crf_train = "crf_learn -f 3 template.txt ../data/train.txt dg_model"
os.system(crf_train)

#4 crf test
crf_test = "crf_test -m dg_model ../data/test.txt -o crf_result.txt"
os.system(crf_test)


def extractTestData(inputPath, outputPath):
    input = open(inputPath, 'r', encoding='utf-8', errors='ignore')
    output = open(outputPath, 'w', encoding='utf-8', errors='ignore')
    for line in input.readlines():
        if len(line.strip()) == 0: output.write(line); continue
        output.write(line.strip().split('\t')[0] + '\n')
    input.close(); output.close()

#extractTestData('../data/valid.txt', 'test.txt')

def caculatorF1Score(realPath, prePath):
    def extractTag(inputPath):
        input = open(inputPath, 'r', encoding='utf-8', errors='ignore')
        tag, tagArr = [], []
        for line in input.readlines():
            line = line.strip('\n')
            if len(line) == 0:
                if len(tag) != 0: tagArr.append(tag); tag = []
                continue
            tag.append(line.split('\t')[1])
        
        if len(tag) != 0: tagArr.append(tag)
        input.close()
        return tagArr

    realArr = extractTag(realPath)
    preArr = extractTag(prePath)

    print (len(realArr), len(preArr))

    f1Score = f1_score(y_true=realArr, y_pred=preArr)
    print (f1Score)

#caculatorF1Score(realPath='../data/valid.txt', prePath='crf_result.txt')

def generateSubmitData(prePath, testLenPath, submitDataPath):
    def extract(inputPath):
        input = open(inputPath, 'r', encoding='utf-8', errors='ignore')
        data = input.read().split('\n\n'); input.close()
        sentenceArr, tagArr = [], []                    
        sentenceArr = [[element2.split('\t')[0] for element2 in element1.split('\n')] for element1 in data if len(element1.strip('\n')) != 0]
        tagArr = [[element2.split('\t')[1] for element2 in element1.split('\n')]for element1 in data if len(element1.strip('\n')) != 0]        
        return sentenceArr, tagArr

    sentenceArr, tagArr = extract(prePath)

    submitData = open(submitDataPath, 'w', encoding='utf-8', errors='ignore')
    testLen = open(testLenPath, 'r', encoding='utf-8', errors='ignore')

    start = 0

    for line in testLen.readlines():
        id, length = line.strip('\n').split('\t')[0], int(line.strip('\n').split('\t')[1])
        sentenceElement, tagElement = sentenceArr[start:start+length], tagArr[start:start+length]
        start += length

        entityArr = acquireEntity(sentenceElement, tagElement)

        #print (entityArr)
            
        def filter_word(w):
            for wbad in ['？','《','🔺','️?','!','#','%','%','，','Ⅲ','》','丨','、','）','（','​',
                    '👍','。','😎','/','】','-','⚠️','：','✅','㊙️','“',')','(','！','🔥',',','.','——', '“', '”', '！', ' ']:
                if wbad in w:
                    return ''
            return w

        #过滤一些无用实体
        entityArr = [entity for entity in entityArr if filter_word(entity) != '' and len(entity) > 1]

        if len(entityArr) == 0: entityArr = ['FUCK']

        submitData.write('%s,%s\n' % (id, ';'.join(entityArr)))
    
    submitData.close(); testLen.close()

generateSubmitData('./crf_result.txt', '../data/test.record', 'crf_submit.csv')
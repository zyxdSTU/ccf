import codecs
import os
from seqeval.metrics import f1_score, accuracy_score, classification_report

#3 crf train
crf_train = "crf_learn -f 5 template.txt ../data/train.txt dg_model"
os.system(crf_train)

# 4 crf test
crf_test = "crf_test -m dg_model test.txt -o crf_result.txt"
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

caculatorF1Score(realPath='../data/valid.txt', prePath='crf_result.txt')
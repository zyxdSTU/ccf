# f = lambda index, arr:[element[index] for element in arr]
# input = open('./data/test.txt', 'r', encoding='utf-8', errors='ignore')
# arr = input.read().split('\n\n')
# #print (arr)
# for item in arr:
#     sentence, tag = [], []
#     #print (item)
#     arrElement = [(element.split('\t')[0], element.split('\t')[1]) for element in item.split('\n') if len(element.strip()) != 0]
#     assert len(f(0, arrElement)) == len(f(1, arrElement))
#     print (f(0,arrElement))
#     print (f(1,arrElement))
# input.close()

# def acquireEntity(sentenceArr, tagArr, config):
#     tagArr = [['B', 'I', 'I', 'O', 'I', 'B', 'I'], ['B', 'I', 'I', 'O']]
#     sentenceArr = [['U', 'U', 'U', 'O', 'F', 'U', 'U'], ['Y', 'Y', 'Y', 'O']]
#     entityArr, entity = [], ''

#     for i in range(len(tagArr)):
#         for j in range(len(tagArr[i])):
#             if tagArr[i][j] == 'B':
#                 if entity != '':entityArr.append(entity); entity = sentenceArr[i][j]
#                 else: entity += sentenceArr[i][j]
#             if tagArr[i][j] == 'I':
#                 if entity != '': entity = entity + sentenceArr[i][j];
#             if tagArr[i][j] == 'O':
#                 if entity != '': entityArr.append(entity); entity = ''

#     if entity != '': entityArr.append(entity)

#     print (entityArr)
#     return list(set(entityArr))

# acquireEntity(None, None, None)

str = 'test\n'
print (str.split('\n'))
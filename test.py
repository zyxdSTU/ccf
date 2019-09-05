f = lambda index, arr:[element[index] for element in arr]
input = open('./data/test.txt', 'r', encoding='utf-8', errors='ignore')
arr = input.read().split('\n\n')
#print (arr)
for item in arr:
    sentence, tag = [], []
    #print (item)
    arrElement = [(element.split('\t')[0], element.split('\t')[1]) for element in item.split('\n') if len(element.strip()) != 0]
    assert len(f(0, arrElement)) == len(f(1, arrElement))
    print (f(0,arrElement))
    print (f(1,arrElement))
input.close()
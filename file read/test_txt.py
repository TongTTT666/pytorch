i = 0
f = open('E:/我的坚果云/Deep Learning/pytorch/codes/test.txt', mode='r')
for line in f:
    print(i)
    i += 1
    print(line)
    line = line.strip('\n')  # 删除开头和结尾的换行符
    line = line.rstrip() # 删除最后的空格
    word = line.split()  # 将每行单词按空格分开
    print(word)
f.close


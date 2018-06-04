import os

FileList  = os.listdir("trainingDigits")#打开训练数据集目录，并获取该目录下的所有文件
TrainSet = []#初始化训练数据集，
label = []#初始化标签集
for File in FileList:
    f = open("trainingDigits/"+File,"r")#打开训练数据集目录中的某一个文件
    numbers = [] #初始化numbers为列表
    content =  f.read()
    for char in content:
        if char != '\n' :#是换行符则忽略 '\n'是换行符
            t = int(char)#将字符转化为数字
            numbers.append(t)
    #经过上述步骤，一个32x32的文件，转变成了一个1024长度的列表，列表中元素为0或者1
    TrainSet.append(numbers) #训练集测试数据+1
    label.append(File[0])#0_0.txt表示该文件是属于0，1_0.txt表示该文件是1，字符串的第0位就是这个文件所代表的数字
    f.close()#关闭文件

f = open("testDigits/"+"0_0.txt","r")#这里我让他们随便制定测试集testDigits中的任意一个文件
TestData = []
content = f.read()
for char in content:
    if char != '\n':
        t = int(char)
        TestData.append(t)
#将文件转化为一个1024长度的列表，并存放在TestData中

Dist = []
for Data in TrainSet:#从训练集中逐渐取出所有数据
    length = len(Data)#获取数据的长度，实际上是1024
    Distance = 0
    for i in range(length):#计算欧氏距离，这个讲过了
        Distance = Distance + (Data[i] - TestData[i])**2
    Dist.append(Distance)#将距离存起来
print(Dist)

length = len(Dist)#冒泡排序
for i in range(length):
    for j in range(length - 1):
        if Dist[j] > Dist[j + 1]:
            tmp = Dist[j]
            Dist[j]  = Dist[j+1]
            Dist[j + 1] = tmp
            tmp = label[j]
            label[j] = label[j + 1]
            label[j + 1] =  tmp

for i in range(length):#排序结果输出
    print("label = ",label[i],"distance = ",Dist[i])

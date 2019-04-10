
# coding: utf-8

# In[5]:

import numpy as np


# In[6]:

#输入数据
X = np.array([[1,0,0],
              [1,0,1],
              [1,1,0],
              [1,1,1]])
#标签
Y = np.array([[0],
              [1],
              [1],
              [0]])



# 3-10-1
#权值初始化，取值范围-1到1
V = np.random.random([3,10])*2-1 
W = np.random.random([10,1])*2-1
print(V)
print(W)
#学习率设置
lr = 0.11

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return x*(1-x)

def update():
    global X,Y,W,V,lr
    
    # 求每一层输出
    L1 = sigmoid(np.dot(X,V))#隐藏层输出 (4,10)
    L2 = sigmoid(np.dot(L1,W))#输出层输出 (4,1)
    
    # 求每一层的学习信号
    L2_delta = (Y - L2)*dsigmoid(L2) #(4,1)
    L1_delta = L2_delta.dot(W.T)*dsigmoid(L1) #(4,10) 
    
    # 求每一层权值的改变
    W_C = lr*L1.T.dot(L2_delta)
    V_C = lr*X.T.dot(L1_delta)
    
    # 改变权值
    W = W + W_C
    V = V + V_C


# In[7]:

for i in range(10000):
    update()#更新权值
    if i%500==0:
        L1 = sigmoid(np.dot(X,V))#隐藏层输出
        L2 = sigmoid(np.dot(L1,W))#输出层输出
        print('loss:',np.mean(np.square(Y-L2)/2))
        
L1 = sigmoid(np.dot(X,V))#隐藏层输出
L2 = sigmoid(np.dot(L1,W))#输出层输出
print(L2)

def judge(x):
    if x>=0.5:
        return 1
    else:
        return 0

for i in map(judge,L2):
    print(i)


# In[ ]:




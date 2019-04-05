
# coding: utf-8

# In[1]:

import numpy as np


# In[2]:

a = np.array([1,2,3],dtype=np.int32)
print(a.dtype)


# In[3]:

b = np.array([1,2,3],dtype=np.float)
print(b.dtype)


# In[4]:

c = np.array([1,2,3])#一维数组
print(c)


# In[5]:

d = np.array([[1,2,3],   #2维矩阵
                      [4,5,6]])
print(d)


# In[6]:

zero = np.zeros((2,3)) #生成2行3列全为0的矩阵
print(zero)


# In[7]:

one = np.ones((3,4)) #生成3行4列全为1的矩阵
print(one)


# In[8]:

empty = np.empty((3,2))#生成3行2列全都接近于0（不等于0）的矩阵
print(empty)


# In[9]:

e = np.arange(10)
print(e)


# In[10]:

f = np.arange(4,12)
print(f)


# In[11]:

g = np.arange(1,20,3)
print(g)


# In[12]:

np.arange(8)


# In[13]:

h = np.arange(8).reshape(4,2)#重新定义矩阵的形状
print(h)


# In[ ]:




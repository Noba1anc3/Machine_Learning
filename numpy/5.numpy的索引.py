
# coding: utf-8

# In[1]:

import numpy as np


# In[2]:

arr1 = np.arange(2,14)
print(arr1)


# In[3]:

print(arr1[2])#第二个位置的数据


# In[4]:

print(arr1[1:4])#第一到第四个位置的数据


# In[5]:

print(arr1[2:-1])#第二到倒数第一个位置的数据


# In[6]:

print(arr1[:5])#前五个数据


# In[7]:

print(arr1[-2:])#最后两个数据


# In[8]:

arr2 = arr1.reshape(3,4)
print(arr2)


# In[9]:

print(arr2[1])


# In[10]:

print(arr2[1][1])


# In[11]:

print(arr2[1,2])


# In[16]:

print(arr2[:,2])


# In[17]:

for i in arr2: #迭代行
    print(i)


# In[18]:

for i in arr2.T:#迭代列
    print(i)


# In[19]:

for i in arr2.flat:#一个一个元素迭代
    print(i)


# In[ ]:





# coding: utf-8

# In[1]:

import numpy as np


# In[2]:

arr1 = np.array([[1,2,3],
                           [4,5,6]])
arr2 = np.array([[1,1,2],
                           [2,3,3]])
print(arr1)
print(arr2)


# In[3]:

print(arr1 + arr2)


# In[4]:

print(arr1 - arr2)


# In[5]:

print(arr1 * arr2)


# In[6]:

print(arr1 ** arr2)


# In[7]:

print(arr1 / arr2)


# In[8]:

print(arr1 % arr2)


# In[9]:

print(arr1 // arr2)


# In[10]:

print(arr1+2)#所有的元素加2


# In[11]:

print(arr1*10)#所有的元素乘以10


# In[12]:

arr3 = arr1 > 3 #判断哪些元素大于3
print(arr3)


# In[13]:

arr4 = np.ones((3,5))
print(arr4) 


# In[14]:

print(arr1)        


# In[15]:

np.dot(arr1,arr4)#矩阵乘法   2*3  3*5


# In[16]:

arr1.dot(arr4)#矩阵乘法


# In[17]:

print(arr1)
print(arr1.T)#矩阵转置
print(np.transpose(arr1))#矩阵转置


# In[ ]:




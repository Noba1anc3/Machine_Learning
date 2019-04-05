
# coding: utf-8

# In[1]:

import numpy as np


# In[2]:

arr1 = np.arange(12).reshape((3,4))
print(arr1)


# In[3]:

arr2,arr3 = np.split(arr1,2,axis=1)#横向分割，分成2份
print(arr2)
print(arr3)


# In[4]:

arr4,arr5,arr6 = np.split(arr1,3,axis=0)#纵向分割，分成3份
print(arr4)
print(arr5)
print(arr6)


# In[5]:

arr2,arr3,arr4 = np.split(arr1,3,axis=1)#横向分割，分成3份
print(arr2)
print(arr3)
print(arr4)


# In[6]:

arr7,arr8,arr9 = np.array_split(arr1,3,axis=1)#横向分割，分成3份，不等分割
print(arr7)
print(arr8)
print(arr9)


# In[7]:

arrv1,arrv2,arrv3 = np.vsplit(arr1,3)#纵向分割
print(arrv1)
print(arrv2)
print(arrv3)


# In[8]:

arrh1,arrh2 = np.hsplit(arr1,2)#横向分割
print(arrh1)
print(arrh2)


# In[ ]:




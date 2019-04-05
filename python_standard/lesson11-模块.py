
# coding: utf-8

# In[1]:

import max #导入了max模块


# In[2]:

a = max.func_max(10,34)
print(a)


# In[3]:

from max import func_max #从max模块导入func_max函数


# In[4]:

a = func_max(11,33)
print(a)


# In[5]:

from max import * #导入max模块中所有的内容


# In[6]:

import max as m #导入max，用as指定max的别名为m



# In[7]:

a = m.func_max(22,44)
print(a)


# In[8]:

import os #导入os模块


# In[9]:

print(os.getcwd()) #获取当前文件所在的路径


# In[11]:

os.mkdir('testdir')


# In[ ]:




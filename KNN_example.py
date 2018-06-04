
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np

# 已知分类的数据
x1 = np.array([3,2,1])
y1 = np.array([104,100,81])
x2 = np.array([101,99,98])
y2 = np.array([10,5,2])
plt.scatter(x1,y1,c='r')
plt.scatter(x2,y2,c='b')

# 未知数据
x = np.array([18])
y = np.array([90])
plt.scatter(x,y,c='k')

plt.show()


# In[ ]:




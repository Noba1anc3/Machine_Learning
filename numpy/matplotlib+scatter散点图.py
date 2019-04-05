
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np


# In[2]:

plt.scatter(np.arange(5),np.arange(5))
plt.show()


# In[4]:

x = np.random.normal(size=500)
y = np.random.normal(size=500)

plt.scatter(x,y,s=50,c='b',alpha=0.5)#alpha透明度

plt.xlim((-2,2))
plt.ylim((-2,2))

#隐藏坐标轴
plt.xticks(())
plt.yticks(())

plt.show()


# In[ ]:




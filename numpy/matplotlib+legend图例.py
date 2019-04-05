
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np


# In[5]:

x = np.linspace(-3,3,100)
y1 = 2*x + 1
y2 = x**2

#xy范围
plt.xlim((-1,2))
plt.ylim((-2,3))

#xy描述
plt.xlabel('I AM X')
plt.ylabel('I AM Y')

l1, = plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')
l2, = plt.plot(x,y2,color='blue',linewidth=5.0,linestyle='-')

#画图例
plt.legend(handles=[l1,l2],labels=['test1','test2'],loc='best')

plt.show()


# In[ ]:





# coding: utf-8

# In[3]:

import matplotlib.pyplot as plt
import numpy as np


# In[4]:

x = np.linspace(-1,1,100)
y1 = 2*x + 1
y2 = x**2

plt.figure()
plt.plot(x,y1)

plt.figure()
plt.plot(x,y2)

plt.show()


# In[5]:

x = np.linspace(-1,1,100)
y1 = 2*x + 1
y2 = x**2

plt.figure()
plt.plot(x,y1)

plt.figure(figsize=(8,5))
plt.plot(x,y2)

plt.show()


# In[4]:

plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')#虚线
plt.plot(x,y2,color='blue',linewidth=5.0,linestyle='-')#实线
plt.show()


# In[ ]:




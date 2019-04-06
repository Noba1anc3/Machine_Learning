# coding: utf-8

# In[2]:

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

# In[5]:

# 读入数据 
data = genfromtxt(r"kmeans.txt",delimiter=' ')
plt.scatter(data[:,0],data[:,1])
plt.show()

# In[9]:

# 训练模型
k = 4
model = KMeans(n_clusters=k).fit(data)

# In[10]:

# 分类中心点坐标
center = model.cluster_centers_
print(center)

# In[11]:

# 预测结果
result = model.predict(data)
print(result)

# In[12]:

# 画出各个数据点，用不同颜色表示分类
color = ['r', 'b', 'g', 'y']
for i,d in enumerate(data):
    plt.scatter(d[0],d[1],c=color[result[i]])

# 画出各个分类的中心点
mark = ['*r', '*b', '*g', '*y']
for i,d in enumerate(center):
    plt.plot(d[0],d[1], mark[i], markersize=20)
    
plt.show()

# In[13]:

model.labels_

# In[ ]:

#肘部法则
#for i in range(2,10)
#loss.append(model_)



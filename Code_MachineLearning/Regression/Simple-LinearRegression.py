
# coding: utf-8

# In[1]:

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# 载入数据
x_data = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9]])
y_data = np.array([[6],[5],[7],[10],[12],[13],[14],[17],[20]])
plt.scatter(x_data,y_data)
plt.show()


# In[5]:

# 创建并拟合模型
model = LinearRegression()
model.fit(x_data, y_data)


# In[6]:

# 预测
x_test = np.array([[6.5]])
model.predict(x_test)


# In[6]:

# 画图
x = [[0],[10]]
y = model.predict(x)
plt.plot(x_data, y_data, 'b.')
plt.plot(x, y, 'k-')
plt.show()


# In[ ]:




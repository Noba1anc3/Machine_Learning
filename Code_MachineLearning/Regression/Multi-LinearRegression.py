
# coding: utf-8

# In[1]:

from numpy import genfromtxt
from sklearn import linear_model


# In[2]:

# 读入数据 
data = genfromtxt(r"Delivery.csv",delimiter=',')
print(data)


# In[3]:

# 切分数据
x_data = data[:,:-1]  #所有行，0列到-1列
y_data = data[:,-1]
print(x_data)
print(y_data)


# In[4]:

# 创建模型
model = linear_model.LinearRegression()
model.fit(x_data, y_data)


# In[5]:

# 系数
print("coefficients:",model.coef_)

# 截距
print("intercept:",model.intercept_)

# 测试
x_test = [[102,4]]
predict = model.predict(x_test)
print("predict:",predict)


# In[7]:

model.score(x_data, y_data)


# In[ ]:





# coding: utf-8

# In[1]:

# 导入算法包以及数据集
from sklearn import neighbors
from sklearn import datasets


# In[2]:

# 载入数据
iris = datasets.load_iris()
print(iris)


# In[3]:

# 构建模型
knn = neighbors.KNeighborsClassifier()
knn.fit(iris.data, iris.target)


# In[10]:

predictLabel = knn.predict([[ 6.7,  3.3,  5.7,  2.5]])
print(predictLabel)


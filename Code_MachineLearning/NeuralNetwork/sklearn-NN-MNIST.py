
# coding: utf-8
#pip install scikit-learn --upgrade
# In[17]:

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# In[18]:

digits = load_digits()#载入数据
x_data = digits.data #数据
y_data = digits.target #标签


print(x_data.shape)
print(y_data.shape)

# In[19]:

plt.imshow(digits.images[0],cmap='gray')
plt.show()


# In[20]:

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data) #分割数据1/4为测试数据，3/4为训练数据


# In[21]:

mlp = MLPClassifier(hidden_layer_sizes=(100,50),max_iter=500)
mlp.fit(x_train,y_train)


# In[7]:

predictions = mlp.predict(x_test)
print(classification_report(y_test, predictions))


# In[ ]:




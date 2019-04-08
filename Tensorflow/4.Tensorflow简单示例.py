
# coding: utf-8

# In[2]:

import tensorflow as tf
import numpy as np


# In[3]:

#使用numpy生成100个随机点
#样本点
x_data = np.random.rand(100)
y_data = x_data*0.1 + 0.2

#构造一个线性模型
d = tf.Variable(1.1)
k = tf.Variable(0.5)
y = k*x_data + d

#二次代价函数
loss = tf.losses.mean_squared_error(y_data,y)
#定义一个梯度下降法来进行训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.3)
#最小化代价函数
train = optimizer.minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20 ==0:
            print(step,sess.run([k,d]))


# In[ ]:




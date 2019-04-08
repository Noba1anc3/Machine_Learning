
# coding: utf-8

# In[1]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[2]:

#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#每个批次的大小
batch_size = 64
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#创建一个简单的神经网络
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#二次代价函数
loss = tf.losses.mean_squared_error(y,prediction)
#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    #epoch：所有数据训练一次，就是一个epoch周期
    for epoch in range(21):
        #batch：一般为32，64个数据
        for batch in range(n_batch):
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))

        
Iter 0,Testing Accuracy 0.8602
Iter 1,Testing Accuracy 0.882
Iter 2,Testing Accuracy 0.8919
Iter 3,Testing Accuracy 0.8962
Iter 4,Testing Accuracy 0.901
Iter 5,Testing Accuracy 0.9042
Iter 6,Testing Accuracy 0.9068
Iter 7,Testing Accuracy 0.9074
Iter 8,Testing Accuracy 0.9092
Iter 9,Testing Accuracy 0.911
Iter 10,Testing Accuracy 0.9122
Iter 11,Testing Accuracy 0.9128
Iter 12,Testing Accuracy 0.9135
Iter 13,Testing Accuracy 0.9143
Iter 14,Testing Accuracy 0.9156
Iter 15,Testing Accuracy 0.9155
Iter 16,Testing Accuracy 0.9167
Iter 17,Testing Accuracy 0.9173
Iter 18,Testing Accuracy 0.9185
Iter 19,Testing Accuracy 0.9181
Iter 20,Testing Accuracy 0.9182


def function3(a=10,b=20): #设置默认值
    c = a + b
    print('a=',a)
    print('b=',b)
    print('c=',c)
    print('a+b=',c)

a = 1000


# In[24]:

def function5(b=20): #设置默认值
    global a   #使用全局变量
    c = a + b
    print('a=',a)
    print('b=',b)
    print('c=',c)
    print('a+b=',c)


# In[25]:

function5()


# In[27]:

# 有返回值的函数
def add(a,b):
    c = a + b
    return c


# In[28]:

d = add(23,45)
print(d)


# In[ ]:




# In[ ]:




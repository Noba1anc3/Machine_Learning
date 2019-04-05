
# coding: utf-8

# In[6]:

class human:  #类
    #类的属性
    name = 'someone' 
    age = 100
    #类的方法
    def my_name(self):
        print('my name is',self.name)
    def my_age(self):
        print('my age is',self.age)
    def eat(self):
        print('eat')
    def think(self,a,b):
        print(a+b)


# In[7]:

person1 = human() #创建一个person1的对象


# In[19]:

class human:  #类
    def __init__(self,name='someone',age=10):#创建对象时会执行
        self.name = name
        self.age = age
        
    #类的方法
    def my_name(self):
        print('my name is',self.name)
    def my_age(self):
        print('my age is',self.age)
    def eat(self):
        print('eat')
    def think(self,a,b):
        print(a+b)



# In[20]:

person3 = human()


# In[16]:

person3.my_name()


# In[17]:

person3 = human(name='xiaohong',age=20)


# In[18]:

person3.my_name()


# In[ ]:




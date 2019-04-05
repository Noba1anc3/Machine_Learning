
# coding: utf-8

# In[1]:

class human:  #类
    def __init__(self,name='someone',age=10):#创建对象时会执行
        self.name = name
        self.age = age
        print('human init')
        
    #类的方法
    def my_name(self):
        print('my name is',self.name)
    def my_age(self):
        print('my age is',self.age)
    def eat(self):
        print('eat')
    def think(self,a,b):
        print(a+b)


# In[2]:

class student(human):#子类继承父类
    pass


# In[3]:

stu1 = student()


# In[4]:

stu1.my_age()


# In[6]:

class student(human):#子类继承父类
    def __init__(self,grade=1,school='MIT'):
        human.__init__(self) #父类的初始化
        self.grade = grade
        self.school = school
        self.scroe = 100
        print('student init')
        
    #添加子类自己的方法
    def learn(self):
        print('learning')
    def my_school(self):
        print('my school is',self.school)


# In[7]:

stu2 = student(4)


# In[8]:

stu2.my_age()


# In[9]:

stu2.learn()


# In[10]:

stu2.my_school()


# In[11]:

class student(human):#子类继承父类
    def __init__(self,grade=1,school='MIT'):
        human.__init__(self) #父类的初始化
        self.grade = grade
        self.school = school
        self.scroe = 100
        print('student init')
        
    #添加子类自己的方法
    def learn(self):
        print('learning')
    def my_school(self):
        print('my school is',self.school)
    #子类可以重写父类的方法
    def think(self,a,b):
        print(a*b)


# In[12]:

stu3 = student()


# In[13]:

stu3.think(10,20)


# In[ ]:




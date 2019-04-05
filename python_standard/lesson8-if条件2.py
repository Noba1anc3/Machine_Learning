
# coding: utf-8

# In[ ]:

'''
and  并且
or   或者
'''


# In[1]:

a = 1
b = 2
c = 3
d = 1

if a<b and a==d:
    print("right")


# In[2]:

if a>b or a==d:
    print("right")


# In[3]:

colors = ['red','blue','black','green']

for color in colors:
    if color == 'black':
        print('black')
    else:
        print('not black')


# In[4]:

for color in colors:
    if color == 'black':
        break           #跳出大循环
        print('black')
    else:
        print('not black')


# In[5]:

for color in colors:
    if color == 'black':
        continue    #跳出单次循环    
        print('black')
    else:
        print('not black')


# In[6]:

if 'red' in colors:#判断列表中是否有'red'，返回值是True False
    print('red')


# In[7]:

null = []
if null: #判断列表是否为空，有值返回True，空的话返回False
    print(1)
else:
    print(0)


# In[1]:

if "ann" < "agg":
    print(1)
else:
    print(0)


# In[ ]:




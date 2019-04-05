
# coding: utf-8

# In[1]:

a_list = [1,2,3,4,5]

d = {'pen':7,'apple':3,'applepen':10} #Key:Value,键：值
d2 = {1:'a',5:'b',2:'d'}
d3 = {1.2:3,'a':3.5,1:'aaa'}

print(d)
print(d2)
print(d3)


# In[2]:

print(d['apple'])


# In[3]:

print(d2[2])


# In[4]:

print(d3[1.2])


# In[5]:

d4 = {'a':[1,2,3,4],'b':(1,2,3,4),'c':{'aa':1,'bb':2}}
print(d4)


# In[6]:

print(d4['c']['aa'])


# In[7]:

d['pen'] = 10 #修改Key对应的Value
print(d)


# In[8]:

d['pineapple'] = 3 #新增一个key：value对
print(d)


# In[9]:

del d['pineapple'] #删除一个键值对
print(d)


# In[10]:

for key,value in d.items():#遍历整个字典的键值对
    print('key:',key,'\t','value:',value)


# In[11]:

for key in d.keys():  #遍历整个字典的键
    print('key:',key)


# In[12]:

for value in d.values(): #遍历整个字典的值
    print('values:',value)


# In[13]:

for key in sorted(d2.keys()):
    print('key:',key)


# In[ ]:




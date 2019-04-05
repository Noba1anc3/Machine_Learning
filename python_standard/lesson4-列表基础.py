
# coding: utf-8

# In[1]:

a_list = [1,2,30,30,30,4,2]#列表
print(a_list)


# In[2]:

print(a_list[1])#打印列表中第1个元素


# In[3]:

print(a_list[5])  #打印列表中第5个元素
print(a_list[-2]) #打印列表中倒数第2个元素


# In[4]:

print(a_list[1:4]) #打印列表中第1到第4个元素,不包括4


# In[5]:

print(a_list[:-2])#从头打印列表直到倒数第2个元素


# In[6]:

print(a_list[-3:])#打印列表后三个元素


# In[7]:

for content in a_list:
    print(content)


# In[8]:

len(a_list)


# In[9]:

print(a_list.index(30))


# In[10]:

print(a_list.count(30))


# In[11]:

a_list.sort()
print(a_list)


# In[12]:

a_list.sort(reverse=True)
print(a_list)


# In[ ]:




# In[ ]:





# coding: utf-8

# In[1]:

a_list = [1,2,30,30,30,4,2]
print(a_list)


# In[2]:

a_list[0] = 100 #修改列表中第0个元素
print(a_list) 


# In[3]:

a_list.append(200) #在列表末尾添加一个元素
print(a_list)


# In[4]:

a_list.insert(2,300) #在列表中插入一个元素
print(a_list)


# In[5]:

del a_list[2] #删除列表第2个元素
print(a_list)


# In[6]:

a_list.remove(30) #删除列表中的一个‘30’
print(a_list)


# In[7]:

a = a_list.pop()
print(a)
print(a_list)


# In[8]:

b_list = [[1,2,3],
             [4,5,6],
             [7,8,9]]
print(b_list[1])


# In[9]:

print(b_list[2][1])


# In[ ]:




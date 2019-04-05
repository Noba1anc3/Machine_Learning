
# coding: utf-8
try:
    file = open('hahaha','r+')
except Exception as e:
    print(e)
    response = input('Do you want to create it:')
    if(response=='yes'):
        with open('hahaha','w') as f:
            pass
        print('The file was created successfully')
    else:
        pass


# In[5]:

try:
    file = open('hahaha','r+')
except Exception as e:
    print(e)
    response = input('Do you want to create it:')
    if(response=='yes'):
        with open('hahaha','w') as f:
            pass
        print('The file was created successfully')
    else:
        pass
else:#没有错误
    file.write('hahaha')
    file.close()


# In[ ]:




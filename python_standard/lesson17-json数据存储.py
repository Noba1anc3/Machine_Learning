
# coding: utf-8

# In[1]:

import json

a_dict = {'user_id':'qbf','user_name':'hello',100:200}

with open('example.json','w') as f:
    json.dump(a_dict,f)


# In[2]:

with open('example.json') as f:
    content = json.load(f)
    print(content)


# In[ ]:




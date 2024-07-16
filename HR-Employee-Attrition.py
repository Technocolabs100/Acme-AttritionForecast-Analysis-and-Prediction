#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np


# In[10]:


df = pd.read_csv("HR-Employee-Attrition.csv")
df


# In[42]:


df.head()


# In[44]:


df.tail()


# In[46]:


df.shape


# In[48]:


df.describe()


# In[50]:


df.columns


# In[52]:


df.nunique()


# In[61]:


df.isnull().sum()


# In[67]:


df.duplicated().sum()


# In[65]:


employee = df.drop(['Over18','EducationField','Department','Attrition','BusinessTravel','Gender'],axis=1)
employee.head()


# In[80]:


df.select_dtypes(include="number").columns


# In[2]:


from sklearn import linear_model
import matplotlib.pyplot as plt


# In[11]:


plt.xlabel('Age')

plt.ylabel('Attrition')
plt.scatter(df.Age,df.Attrition,color='red',marker='+')


# In[ ]:





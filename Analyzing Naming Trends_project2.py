#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


from io import BytesIO

from zipfile import ZipFile


# In[6]:


z = ZipFile('names.zip').extractall('.')


# In[12]:





# In[13]:


years = []
 
for year in range(1880, 2024):
    years.append(pd.read_csv(f'yob{year}.txt', names = ['Name', 'Sex', 'Babies']))
    years[-1]['Year'] = year
     


# In[27]:





# In[14]:


df = pd.concat(years)
df


# In[16]:


#df= pd.read_csv(BytesIO(z.read('names.zip')), encoding='utf-8',engine='python', header=None)


# In[15]:


df.info()


# In[29]:


#We have to extract the dataset in the program,
#visualize the number of male and female babies born in a particular year, and

male_female_babies_counts= df.groupby(['Sex', 'Year'])['Babies'].sum().reset_index()
male_female_babies_counts


# In[33]:


female_countyearwise=male_female_babies_counts[(male_female_babies_counts['Sex']=='F')]
female_countyearwise


# In[32]:


male_countsyearwise=male_female_babies_counts[(male_female_babies_counts['Sex']=='M')]
male_countsyearwise


# In[46]:


df.duplicated().sum()


# In[47]:


df.isnull().sum()


# In[52]:


most_popular_name=df[['Year','Name','Sex','Babies']][df.values==df['Babies'].max()]
most_popular_name


# In[56]:


#find out popular baby names.
#where we are having more number of babies of same name that will be popular name.
#lets find out yearwise
#for 1880 FEMALE
year1880_F=df[(df['Year']==1880)&(df['Sex']=='F')]
year1880_F_popular_name=year1880_F[['Name','Babies']][year1880_F.values==year1880_F['Babies'].max()]
year1880_F_popular_name


# In[63]:


#male in 1880
year1880_M=df[(df['Year']==1880)&(df['Sex']=='M')]
year1880_M_popular_name=year1880_M[['Name','Babies']][year1880_M.values==year1880_M['Babies'].max()]
year1880_M_popular_name


# In[59]:


#FOR YEAR 1993
#MALE
year1993_M=df[(df['Year']==1993)&(df['Sex']=='M')]
year1993_M_popular_name=year1993_M[['Name','Babies']][year1993_M.values==year1993_M['Babies'].max()]
year1993_M_popular_name


# In[60]:


#FEMALE IN 1993
year1993_F=df[(df['Year']==1993)&(df['Sex']=='F')]
year1993_F_popular_name=year1993_F[['Name','Babies']][year1993_F.values==year1993_F['Babies'].max()]
year1993_F_popular_name


# In[61]:


#FOR 2023
#male
year2023_M=df[(df['Year']==2023)&(df['Sex']=='M')]
year2023_M_popular_name=year2023_M[['Name','Babies']][year2023_M.values==year2023_M['Babies'].max()]
year2023_M_popular_name


# In[62]:


#for female in 2023
year2023_F=df[(df['Year']==2023)&(df['Sex']=='F')]
year2023_F_popular_name=year2023_F[['Name','Babies']][year2023_F.values==year2023_F['Babies'].max()]
year2023_F_popular_name


# In[ ]:





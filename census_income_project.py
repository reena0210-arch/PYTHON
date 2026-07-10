#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[34]:


df =pd.read_csv('census-income (7).csv')
df


# In[35]:


df.info()


# In[36]:


df['income']=df.iloc[:,-1]


# In[37]:


df.drop(columns=[' '],inplace=True)


# In[38]:


df.info()


# In[39]:


df.isnull().sum()


# In[40]:


df.duplicated().sum()


# In[41]:


df.drop_duplicates(inplace=True)


# In[42]:


df.duplicated().sum()


# In[43]:


df.info()


# In[44]:


#visuvalization part
sns.lineplot(x=df['age'],y=df['income'])
plt.show()


# In[52]:


df.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain',
             'capital-loss','hours-per-week','native-country','income']


# In[53]:


df.info()


# In[56]:


sns.lineplot(x=df['workclass'],y=df['income'])
plt.show()


# In[57]:


#data cleaning
df['workclass'].unique()


# In[59]:


(df['workclass'].mode())


# In[60]:


df['workclass'] =df['workclass'].str.replace(' ','')
df['workclass'] =df['workclass'].replace('?','Private')


# In[61]:


df['education'].unique()


# In[63]:


df['education']=df['education'].str.replace(' ','')
df['education'].unique()


# In[64]:


df['education-num'].unique()


# In[65]:


df['capital-gain'].unique()


# In[66]:


df['native-country'].unique()


# In[67]:


(df['native-country']==' ?').value_counts()


# In[68]:


df['native-country'].mode()


# In[69]:


df['native-country']=df['native-country'].str.replace(' ','')
df['native-country']=df['native-country'].replace('?','United-States')


# In[70]:


df['marital-status'].unique() 


# In[71]:


df.describe()


# In[72]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in df.columns:
    if df[col].dtype=='object':
        df[col]=le.fit_transform(df[col])


# In[75]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
x=df.iloc[:,:-1]
vif_data=pd.DataFrame()
vif_data['features']=x.columns
vif_data['VIF']=[variance_inflation_factor(x.values,i)for i in range(len(x.columns))]
vif_data


# In[76]:


#vif value is greater then 5 for age,workclass,education,education_num,race,hours-per-week,native-country
#so lets drop these columns.


# In[79]:


df.drop(columns=['age','workclass','race','hours-per-week','native-country','education','education-num'],inplace=True)


# In[80]:


df.info()


# In[81]:


x=df.iloc[:,:-1]
y=pd.DataFrame(df.iloc[:,-1])


# In[86]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=20,test_size=0.30)


# In[93]:


#lest check with linear model
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)


# In[94]:


from sklearn.metrics import *
score=accuracy_score(y_test,y_pred)
print(f'score from logistic model:{score}')


# In[98]:


from sklearn.tree import DecisionTreeClassifier
dtr=DecisionTreeClassifier(max_depth=6)
dtr.fit(x_train,y_train)
y_pred=dtr.predict(x_test)


# In[99]:


from sklearn.metrics import *
score2=accuracy_score(y_test,y_pred)
print(f'score from decisiontree model:{score2}')


# In[100]:


from sklearn.tree import plot_tree
plot_tree(dtr,filled=True)
plt.show()


# In[101]:


from sklearn.ensemble import RandomForestClassifier
rfr=RandomForestClassifier(max_depth=6)
rfr.fit(x_train,y_train)
y_pred=dtr.predict(x_test)


# In[102]:


from sklearn.metrics import *
score3=r2_score(y_test,y_pred)
print(f'score from radom forest model:{score2}')


# In[ ]:


#so decision tree classification and random forest regression accuracy is 84.68%


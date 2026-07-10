#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[44]:


bf= pd.read_csv('smoking.csv')
bf


# In[45]:


bf.info()


# In[46]:


bf.duplicated().sum()


# In[47]:


bf.drop(columns=['ID'],inplace=True)


# In[48]:


bf.isnull().sum()


# In[49]:


bf['gender'].unique()


# In[50]:


#changing dtype 
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in bf.columns:
    if bf[col].dtype=='object':
        bf[col]=le.fit_transform(bf[col])


# In[51]:


bf.info()


# In[56]:


#check for outliers
for col in bf.columns:
    sns.boxplot(bf[col])
    plt.title(col)
    plt.show()


# In[59]:


for col in bf.columns:
    q1= bf[col].quantile(0.25)
    q3=bf[col].quantile(0.75)
    iqr=q3-q1
    lower_limit=q1-1.5*iqr
    upper_limit=q3+1.5*iqr
    bf=bf[(bf[col]>=lower_limit) &bf[col]<=upper_limit]
        


# In[60]:


bf


# In[61]:


y= pd.DataFrame(bf.iloc[:,-1])
x = bf.iloc[:,:-1]


# In[63]:


from sklearn.ensemble import ExtraTreesClassifier
et=ExtraTreesClassifier()
model=et.fit(x,y)
df = pd.Series(model.feature_importances_,index=x.columns)
df.nlargest(25).plot(kind='barh')
plt.show()


# In[ ]:


#lets select top 12 columns
x=bf[['gender','height(cm)','Gtp','triglyceride','weight(Kg)','age','waist(cm)','HDL','fasting blood sugar','ALT','LDL']]


# In[64]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=30,test_size=0.20)


# In[65]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)


# In[67]:


from sklearn.metrics import *
accuracy_score=accuracy_score(y_test,y_pred)
accuracy_score


# In[68]:


cr= classification_report(y_test,y_pred)
cr


# In[70]:


cn=confusion_matrix(y_test,y_pred)
print(cn)
sns.heatmap(cn,annot=True)
plt.show()


# In[71]:


#logistic regression model accuracy is 72%
#lets check from decision tree
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred= dt.predict(x_test)


# In[72]:


cn=confusion_matrix(y_test,y_pred)
cn


# In[75]:


from sklearn.metrics import *
a_s2= accuracy_score(y_test,y_pred)
a_s2


# In[76]:


#decision tree classifier accuracy is 79.5%


# In[78]:


#Bagging Algorithm – Bagging Classifier
from sklearn.ensemble import BaggingClassifier
bc=BaggingClassifier(estimator=DecisionTreeClassifier(),n_estimators=500)
bc.fit(x_train,y_train).score(x_test,y_test)
y_pred=bc.predict(x_test)



# In[79]:


from sklearn.metrics import *
classification_report(y_test,y_pred)


# In[80]:


as3=accuracy_score(y_test,y_pred)
as3


# In[82]:


#Bagging Algorithm – Extra Trees
from sklearn.ensemble import ExtraTreesClassifier
et=ExtraTreesClassifier(n_estimators=500,random_state=33)
et.fit(x_train,y_train)
y_pred=et.predict(x_test)


# In[83]:


from sklearn.metrics import *
classification_report(y_test,y_pred)


# In[84]:


as4=accuracy_score(y_test,y_pred)
as4


# In[85]:


#Bagging Algorithm – Random Forest
from sklearn.ensemble import RandomForestClassifier
rt=RandomForestClassifier(n_estimators=500,random_state=30)
rt.fit(x_train,y_train)
y_pred=rt.predict(x_test)


# In[86]:


as5=accuracy_score(y_test,y_pred)
as5


# In[ ]:





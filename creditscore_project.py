#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df= pd.read_csv('credit_score.csv',low_memory=False)
df


# In[3]:


df.info()


# In[4]:


df.duplicated().sum()


# In[5]:


df.isnull().sum()


# In[6]:


df1=df.drop(columns=['ID','Customer_ID','Name','SSN'])
df1


# In[7]:


df1.info()


# In[ ]:





# In[8]:


df1['Age']=df1['Age'].str.replace('_','')
df1['Age']=df1['Age'].str.replace('-','')
df1['Age']=df1['Age'].astype(int)


# In[9]:


df1['Occupation'] =df1['Occupation'].replace('_______',np.nan)


# In[10]:


df1['Annual_Income']=df1['Annual_Income'].str.replace('_','')
df1['Annual_Income']=df1['Annual_Income'].astype(float)


# In[11]:


df1['Num_of_Loan']=df1['Num_of_Loan'].str.replace('_','')
df1['Num_of_Loan']=df1['Num_of_Loan'].astype(int)


# In[ ]:





# In[12]:


df1['Type_of_Loan']


# In[13]:


#df1['Delay_from_due_date'].unique()


# In[14]:


df1['Delay_from_due_date']=df1['Delay_from_due_date'].replace('-',np.nan)


# In[15]:


#df1['Num_of_Delayed_Payment'].unique()


# In[16]:


df1['Num_of_Delayed_Payment']= df1['Num_of_Delayed_Payment'].str.replace('_','')
df1['Num_of_Delayed_Payment']=df1['Num_of_Delayed_Payment'].str.replace('-','')

df1['Num_of_Delayed_Payment'] = df1['Num_of_Delayed_Payment'].astype(float)


# In[17]:


#df1['Changed_Credit_Limit'].unique()
df1['Changed_Credit_Limit']=df1['Changed_Credit_Limit'].replace('',np.nan)
df1['Changed_Credit_Limit']=df1['Changed_Credit_Limit'].replace('_',np.nan)

df1['Changed_Credit_Limit'] = df1['Changed_Credit_Limit'].astype(float)


# In[18]:


df1['Credit_Mix']= df1['Credit_Mix'].replace("_",np.nan)
df1['Credit_Mix'] =df1['Credit_Mix'].replace(['Good', 'Standard', 'Bad'],[0,1,2])


# In[19]:


df1['Credit_Mix'].unique()


# In[20]:


df1['Outstanding_Debt']=df1['Outstanding_Debt'].replace('_','')
df1['Outstanding_Debt']=df1['Outstanding_Debt'].str.replace('_','')
df1['Outstanding_Debt']=df1['Outstanding_Debt'].astype(float)


# In[21]:


#df1['Payment_of_Min_Amount'].unique()
df1['Payment_of_Min_Amount']=df1['Payment_of_Min_Amount'].replace(['No', 'NM', 'Yes'],[0,0,1])


# In[22]:


df1['Amount_invested_monthly']=df1['Amount_invested_monthly'].str.replace('__','')
df1['Amount_invested_monthly']=df1['Amount_invested_monthly'].astype(float)


# In[23]:


df1['Payment_Behaviour']=df1['Payment_Behaviour'].replace('!@9#%8',np.nan)


# In[24]:


df1['Monthly_Balance']=df1['Monthly_Balance'].str.replace('__','')
df1['Monthly_Balance']=df1['Monthly_Balance'].str.replace('-','')
df1['Monthly_Balance']=df1['Monthly_Balance'].astype(float)


# In[25]:


df1['Credit_Score']= df1['Credit_Score'].replace(['Good', 'Standard', 'Poor'],[0,1,2])


# In[26]:


df1.info()


# In[27]:


df1.isnull().sum()


# In[28]:


df1.ffill(axis=0,inplace=True,limit=None)
df1.bfill(axis=0,inplace=True,limit=None)


# In[29]:


df1.isnull().sum()


# In[30]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in df1.columns:
    if df1[col].dtype=='object':
        df1[col]=le.fit_transform(df1[col])
        
        


# In[31]:


df1.info()


# In[32]:


#lets remove outliers
sns.boxplot(df1['Age'])
plt.title('Age')
plt.show()


# In[33]:


Q1= df1['Age'].quantile(0.25)
Q3= df1['Age'].quantile(0.75)
iqr=Q3-Q1
upper_limit= Q3+1.5*iqr
lower_limit=Q1-1.5*iqr
df2=df1[(df1['Age'] >lower_limit) &(df1['Age'] <upper_limit)]


# In[34]:


sns.boxplot(df2['Age'])


# In[35]:


df2.info()


# In[36]:


# now lets go for feature selection
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[37]:


X=df2.iloc[:,:-1]
vif_data = pd.DataFrame()
vif_data['feature']=X.columns
vif_data['VIF']=[variance_inflation_factor(X.values,i)for i in range(len(X.columns))]


# In[38]:


vif_data


# In[39]:


#selecting all the feature because vif values is less then 5 for all.


# In[40]:


#logistic regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[42]:


y= pd.DataFrame(df2.iloc[:,-1])
y


# In[43]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=2, test_size=0.30)


# In[ ]:





# In[44]:


model= lr.fit(x_train,y_train)
y_pred = model.predict(x_test)


# In[45]:


from sklearn.metrics import *
accuracy_score=accuracy_score(y_test,y_pred)
accuracy_score


# In[46]:


#decision tree model
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
model= dt.fit(x_train,y_train)
y_pred = model.predict(x_test)


# In[47]:


from sklearn.metrics import *
acc_score2= accuracy_score(y_test,y_pred)
print(f'acc_score2:{acc_score2}')


# In[48]:


#from sklearn.tree import plot_tree
#plot_tree(model)
#plt.show()


# In[49]:


#random forest model
from sklearn.ensemble import RandomForestClassifier
rf  = RandomForestClassifier()


# In[49]:


#parameters={'max_features':['log2','sqrt','auto'],'criterion':['ginni','entropy'],'max_depth':[5,10,15,20,25]
#        }


# In[50]:


#from sklearn.model_selection import GridSearchCV
#model_cv = GridSearchCV(rf,parameters,cv=5)


# In[51]:





# In[50]:


rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)


# In[54]:


accuary_score=accuracy_score(y_test,y_pred)
accuary_score


# In[ ]:





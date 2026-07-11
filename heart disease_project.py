#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df= pd.read_csv('dataset.csv')
df


# In[3]:


df.info()


# In[5]:


#Data Analysis:
#a. Import the dataset
#b. Get information about the dataset (mean, max, min, quartiles etc.)
df.describe()


# In[13]:


#c. Find the correlation between all fields
plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),annot=True)


# In[15]:


#ving a heart disease and not having
#a.visulalize the number of patients having a heart disease and not having heart diseases
df.info()


# In[30]:


labels=['Yes','No']
count_classes = df.value_counts(df['target'], sort= True)
count_classes.plot(kind = "bar", rot = 0)
plt.ylabel("Count")
plt.xticks(range(2),labels)


# In[38]:


#b. Visualize the age and whether a patient has disease or not
plt.figure(figsize=(5,8))
plt.scatter(df['target'],df['age'])

plt.xticks(df['target'])
plt.yticks(df['age'])
plt.show()


# In[41]:


#c. Visualize correlation between all features using a heat map
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True)
plt.show()


# In[42]:


#logistic regression
#a. Build a simple logistic regression model:
#i. Divide the dataset in 70:30 ratio
#ii. Build the model on train set and predict the values on test set
#iii. Build the confusion matrix and get the accuracy score
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[43]:


x= df.iloc[:,:-1]
y=pd.DataFrame(df['target'])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,random_state=42,test_size=0.30) 


# In[75]:


lr.fit(x_train,y_train)
y_pred= lr.predict(x_test)


# In[45]:


from sklearn.metrics import*
acc_score=accuracy_score(y_test,y_pred)
acc_score


# In[46]:


cn= confusion_matrix(y_test,y_pred)
cn


# In[76]:


CR= classification_report(y_test,y_pred)
print(f'CR for logistic regression:{CR}')


# In[47]:


sns.heatmap(cn,annot=True)
plt.show()


# In[48]:


#4. Decision Tree:
#a. Build a decision tree model:
#i. Divide the dataset in 70:30 ratio
#ii. Build the model on train set and predict the values on test set
#iii. Build the confusion matrix and calculate the accuracy
#iv. Visualize the decision tree using the Graphviz package
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(max_depth=5)
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)


# In[49]:


from sklearn.metrics import *
acc2=accuracy_score(y_test,y_pred)
print(acc2)


# In[50]:


cn= confusion_matrix(y_test,y_pred)
print(cn)


# In[58]:


from sklearn.tree import plot_tree
plot_tree(dt,filled=True)
plt.show()


# In[72]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(max_depth=5)
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)


# In[61]:


from sklearn.metrics import *
acc3=accuracy_score(y_test,y_pred)
print(acc3)


# In[62]:


from sklearn.tree import plot_tree
plot_tree(dt,filled=True)
plt.show()


# In[73]:


CR=classification_report(y_test,y_pred)
print(f'CR for decision tree :{CR}')


# In[53]:


from sklearn.tree import export_graphviz
export_graphviz(dt, out_file="dt.dot",impurity=False, filled=True)


# In[56]:


import graphviz
with open("dt.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# In[63]:


#5. Random Forest:
#a. Build a Random Forest model:
#i. Divide the dataset in 70:30 ratio
#ii. Build the model on train set and predict the values on test set
#iii. Build the confusion matrix and calculate the accuracy
#iv. Visualize the model using the Graphviz
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred= rf.predict(x_test)


# In[64]:


from sklearn.metrics import *
acc3= accuracy_score(y_test,y_pred)
print(acc3)


# In[68]:


cn=confusion_matrix(y_test,y_pred)
print(f'confusion_martrix for random forest:{cn}')
sns.heatmap(cn,annot=True)
plt.show()


# In[70]:


CR=classification_report(y_test,y_pred)
print(f'CR for random forest:{CR}')


# In[77]:


#here accuracy of logistic regression is 81% and random forest is 80.21%. so logistic 
#regression and random forest  model both are the best model.


# In[ ]:





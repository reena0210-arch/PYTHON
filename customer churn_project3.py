#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('customer_churn.csv')
df.info()


# In[3]:


df


# In[5]:


df['TotalCharges'] =df['TotalCharges'].astype(float)


# In[6]:


df['TotalCharges'] = df['TotalCharges'].replace(' ','0')


# In[7]:


df['TotalCharges'] =df['TotalCharges'].astype(float)


# In[8]:


df.info()


# In[9]:


df


# In[10]:


#extract the 5th column and store it in ‘customer_5’
customer_5=df['Dependents']
customer_5


# In[11]:


#extract the 15th column and store it in ‘customer_15’
customer_15=df['StreamingMovies']
customer_15


# In[12]:


#extract all the male senior citizens whose payment method is electronic
#check and store the result in ‘senior_male_electronic’.
senior_male_electronic= df[(df['gender']=='Male') & (df['SeniorCitizen']==1) &(df['PaymentMethod']=='Electronic check')]
senior_male_electronic


# In[13]:


#Extract all those customers whose tenure is greater than 70 months or their monthly charges 
#is more than $100 and store the result in ‘customer_total_tenure’.


# In[14]:


customer_total_tenure=df[(df['TotalCharges']>100) &(df['tenure']>70)]
customer_total_tenure


# In[15]:


#Extract all the customers whose contract is of two years, payment method
#is mailed check and the value of churn is ‘Yes’ and store the result in
#‘two_mail_yes’
two_mail_yes=df[(df['Contract']=='Two year') & (df['PaymentMethod']=='Mailed check') & (df['Churn']=='Yes')]
two_mail_yes


# In[16]:


#Extract 333 random records from the customer_churndataframe and store the result in ‘customer_333’
customer_333 =df.sample(n=133)
customer_333


# In[17]:


#Get the count of different levels from the ‘Churn’ column
df['Churn'].value_counts()


# In[18]:


#2. Data Visualization:
#● Build a bar-plot for the ’InternetService’ column:
#a. Set x-axis label to ‘Categories of Internet Service’
#b. Set y-axis label to ‘Count of Categories
##c. Set the title of plot to be ‘Distribution of Internet Service’
#d. Set the color of the bars to be ‘orange'
plt.figure(figsize=(3,5))
plt.bar(df['InternetService'].unique(),df['InternetService'].value_counts(),color='orange',width=0.2)
plt.show()
plt.xlabel('Categories of Internet Service')
plt.ylabel('Distribution of Internet Service')
plt.title('Distribution of Internet Service')
plt.legend('best')


# In[19]:


#Build a histogram for the ‘tenure’ column:
#a. Set the number of bins to be 30
#b. Set the color of the bins to be ‘green’
#c. Assign the title ‘Distribution of tenure
plt.hist(df['tenure'],bins=[0,30,60,90,120,150,180,210])
plt.xticks([0,30,60,90,120,150,180,210])
plt.yticks([500,1000,1500,2000,2500,3000,3500,4000])
plt.grid(True)
plt.show()


# In[20]:


#Build a scatter-plot between ‘MonthlyCharges’ and ‘tenure’. Map‘MonthlyCharges’ to the y-axis 
#and ‘tenure’ to the ‘x-axis’:
#a. Assign the points a color of ‘brown’
#b. Set the x-axis label to ‘Tenure of customer’
#c. Set the y-axis label to ‘Monthly Charges of customer’
#d. Set the title to ‘Tenure vs Monthly Charges’
#e. Build a box-plot between ‘tenure’ & ‘Contract’. Map ‘tenure’ on the y-axis &
#f. ‘Contract’ on the x-axis.


# In[21]:


plt.scatter(df['tenure'],df['MonthlyCharges'],color='brown')
plt.xlabel('Tenure of customer')
plt.ylabel('Monthly Charges of customer')
plt.title('Tenure vs Monthly Charges')


# In[22]:


box_plot= df['tenure']
sns.boxplot(box_plot,showmeans=True)
plt.grid(True)
plt.show()


# In[23]:


box_plot2=df['Contract']
sns.boxplot(box_plot2,showmeans=True)
plt.grid(True)
plt.show()


# In[24]:


#3. linear regression: build a simple linear model where dependent variable is'monthly charges'
#independent variable is ‘tenure’:
#a. Divide the dataset into train and test sets in 70:30 ratio.
#b. Build the model on train set and predict the values on test set
#c. After predicting the values, find the root mean square error
#d. Find out the error in prediction & store the result in ‘error’
#e. Find the root mean square error.


# In[25]:


x= pd.DataFrame(df['MonthlyCharges'])
y=pd.DataFrame(df['tenure'])
x


# In[26]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=30,test_size=0.30)


# In[27]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
model = lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)


# In[28]:


from sklearn.metrics import *
r2_score=r2_score(y_pred,y_test)
r2_score


# In[29]:


rmse= np.sqrt(mean_squared_error(y_pred,y_test))
rmse


# In[30]:


#4. Logistic Regression:
#● Build a simple logistic regression model where dependent variable is
#‘Churn’ and independent variable is ‘MonthlyCharges’:
#a. Divide the dataset in 65:35 ratio
#b. Build the model on train set and predict the values on test set
#c. Build the confusion matrix and get the accuracy score


# In[31]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Churn']=le.fit_transform(df['Churn'])


# In[32]:


x= pd.DataFrame(df['MonthlyCharges'])
y= pd.DataFrame(df['Churn'])

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=30,test_size=0.35)


# In[33]:


from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()


# In[34]:


model= LR.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[35]:


accuracy_score=accuracy_score(y_test,y_pred)
accuracy_score


# In[37]:


cn= confusion_matrix(y_test,y_pred)
print(cn)
sns.heatmap(cn,annot=True)
plt.show()


# In[38]:


df.info()


# In[47]:


#d. Build a multiple logistic regression model where dependent variable
#is ‘Churn’ and independent variables are ‘tenure’ and
#‘MonthlyCharges’
#e. Divide the dataset in 80:20 ratio
#f. Build the model on train set and predict the values on test set
#g. Build the confusion matrix and get the accuracy score
x= pd.DataFrame(df[['tenure','MonthlyCharges']])
y= pd.DataFrame(df['Churn'])


# In[48]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=20,test_size=0.20)


# In[49]:


from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()


# In[50]:


model=LR.fit(x_train,y_train)
y_pred = model.predict(x_test)


# In[54]:


from sklearn.metrics import *
accuracy_score=accuracy_score(y_test,y_pred)
accuracy_score


# In[56]:


cn= confusion_matrix(y_test,y_pred)
print(cn)


# In[57]:


sns.heatmap(cn,annot=True)
plt.show()


# In[58]:


#5. Decision Tree:
#● Build a decision tree model where dependent variable is ‘Churn’ and
#independent variable is ‘tenure’:
#a. Divide the dataset in 80:20 ratio
#b. Build the model on train set and predict the values on test set
#c. Build the confusion matrix and calculate the accuracy


# In[112]:


x= pd.DataFrame(df['tenure'])
y=pd.DataFrame(df['Churn'])
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=67,test_size=0.20)


# In[114]:


from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier(max_depth=8)
model1=dt.fit(x_train,y_train)
y_pred=model1.predict(x_test)


# In[ ]:





# In[115]:


from sklearn.metrics import *
accuracy_score=accuracy_score(y_test,y_pred)
accuracy_score


# In[116]:


cn=confusion_matrix(y_test,y_pred)
cn


# In[100]:


sns.heatmap(cn,annot=True)
plt.show()


# In[101]:


from sklearn.tree import plot_tree
plot_tree(model1,filled=True)
plt.show()



# In[66]:


#6. Random Forest:
#● Build a Random Forest model where dependent variable is ‘Churn’ and
#independent variables are ‘tenure’ and ‘MonthlyCharges’:
#a. Divide the dataset in 70:30 ratio
#b. Build the model on train set and predict the values on test set
#c. Build the confusion matrix and calculate the accuracy


# In[78]:


y = pd.DataFrame(df['Churn'])
x=pd.DataFrame(df[['tenure','MonthlyCharges']])


# In[79]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=20,test_size=0.30)


# In[102]:


param_grid={
    'n_estimators':[50,100,150],
    'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    'min_samples_split':[2,5,7],
    'min_samples_leaf':[1,2,4],
    'criterion':['gini','entropy']
    
    
}


# In[103]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
model2=RandomForestClassifier(random_state=33)
modelgrid=GridSearchCV(estimator=model2,param_grid=param_grid,cv=5,scoring='accuracy')
modelgrid.fit(x_train,y_train)
y_pred=modelgrid.predict(x_test)


# In[104]:


modelgrid.best_params_


# In[106]:


modelgrid.fit(x_train,y_train)


# In[107]:


y_pred= modelgrid.predict(x_test)


# In[108]:


from sklearn.metrics import *
accuracy_score= accuracy_score(y_test,y_pred)
accuracy_score


# In[109]:


cn=confusion_matrix(y_test,y_pred)
cn


# In[110]:


sns.heatmap(cn,annot=True)
plt.show()


# In[ ]:





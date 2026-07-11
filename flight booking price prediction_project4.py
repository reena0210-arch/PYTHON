#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df= pd.read_csv('Flight_Booking.csv')
df


# In[5]:


df.info()


# In[6]:


df.drop('Unnamed: 0',axis=1,inplace=True)


# In[7]:


df.shape


# In[8]:


df.describe()


# In[9]:


df.info()


# In[10]:


df.isnull().sum()


# In[11]:


df.duplicated().sum()


# In[12]:


#visuvalization
#price vs flight
plt.figure(figsize=(10,12))
sns.lineplot(x=df['airline'],y=df['price'])
plt.xlabel('airline')
plt.ylabel('price')
plt.title(' flight vs price')
plt.show()


# In[13]:


#
plt.figure(figsize=(10,12))
sns.lineplot(x=df['days_left'],y=df['price'])
plt.xlabel('days_left')
plt.ylabel('price')
plt.title('days_left vs price')
plt.show()


# In[14]:


#Price range of all the flights
sns.barplot(x=df['airline'],y=df['price'],color='orange')
plt.show()


# In[15]:


#Range of price of all the flights of Economy and Business class
plt.figure(figsize=(5,6))
sns.barplot(x=df['class'],y=df['price'],color='green',hue=df['airline'])
plt.show()


# In[16]:


#Range of price of flights with source and destination city according to the days left.
plt.subplot(1,2,1)
sns.lineplot(x=df['days_left'],y=df['price'],hue=df['source_city'])
plt.title('days left vs price with source-city')
plt.show()



# In[17]:


plt.subplot(1,2,2)
sns.lineplot(x=df['days_left'],y=df['price'],hue=df['destination_city'])
plt.title('days left vs price with destination-city')
plt.show()


# In[18]:


df


# In[19]:


plt.subplot(4,2,1)
plt.hist(df['airline'])
plt.title('airline with its frequency')
plt.show()

plt.subplot(4,2,2)
plt.hist(df['source_city'])
plt.title('source_city with its frequency')
plt.show()

plt.subplot(4,2,3)
plt.hist(df['departure_time'])
plt.title('departure_time with its frequency')
plt.show()

plt.subplot(4,2,4)
plt.hist(df['stops'])
plt.title('stops with its frequency')
plt.show()

plt.subplot(4,2,5)
plt.hist(df['arrival_time'])
plt.title('arrival_time with its frequency')
plt.show()

plt.subplot(4,2,6)
plt.hist(df['class'])
plt.title('class with its frequency')
plt.show()

plt.subplot(4,2,7)
plt.hist(df['destination_city'])
plt.title('destination_city with its frequency')
plt.show()


# In[20]:


df.info()


# In[21]:


#Performing One Hot Encoding for categorical features of a dataframe
from sklearn.preprocessing import LabelEncoder



# In[24]:


le=LabelEncoder()
for col in df.columns:
    if (df[col].dtype=='object') & (col!='flight'):
        df[col]=le.fit_transform(df[col])


# In[25]:


df.info()


# In[44]:


df.drop(columns='flight',inplace=True)


# In[45]:


df.info()


# In[50]:


#Plotting the correlation graph to see the correlation between features and dependent variable.
sns.heatmap(df.corr(),annot=True)
plt.show()


# In[ ]:





# In[53]:


#Selecting the features using VIF. VIF should be less than 5. So drop the stops feature.
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[79]:


col_list=[]
for col in df.columns:
    if (df[col].dtype!='object') & (col!='price'):
        col_list.append(col) 
col_list


# In[80]:


X= df[col_list]
vif_data=pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF']=[variance_inflation_factor(X.values,i) for i in range(len(X.columns))]
print(vif_data)


# In[ ]:





# In[40]:


#Dropping the stops column.All features are having VIF less than 5.


# In[83]:


#Applying standardization and implementing Linear Regression Model to predict the price of a flight.
x= df.iloc[:,:-1]
y= pd.DataFrame(df['price'])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=20,test_size=0.30)


# In[87]:


from sklearn.linear_model import LinearRegression
Lr=LinearRegression()
model=Lr.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[88]:


from sklearn.metrics import *
r2_Score=r2_score(y_test,y_pred)
print(r2_Score)


# In[89]:


from sklearn import metrics
mean_absolute_error=metrics.mean_absolute_error(y_test,y_pred)
print(f'mean_absolute_error:{mean_absolute_error}')
mean_absolute_percentage_error=mean_absolute_percentage_error(y_test,y_pred)
print(f'mean_absolute_percentage_error:{mean_absolute_percentage_error}')
rmse= np.sqrt(mean_squared_error(y_test,y_pred))
print(f'rmse:{rmse}')
#Calculating r2 score,MAE, MAPE, MSE, RMSE. Root Mean square error(RMSE)
#of the Linear regression model is 4623 and Mean absolute percentage
#error(MAPE) is 43 percent. Lower the RMSE and MAPE better the model.


# In[91]:


#Plotting the graph of actual and predicted price of flight
sns.distplot(y_test,label='actual')
sns.distplot(y_pred,label='predicted')
plt.legend()
plt.show()


# In[94]:


#Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
model=dt.fit(x_train,y_train)
y_pred=model.predict(x_test)
r2_score2=r2_score(y_test,y_pred)
print(f'decisiontree r2 score:{r2_score2}')

#Mean absolute percentage error is 7.8 percent and RMSE is 3464 which is less than the linear
#regression model


# In[96]:


from sklearn.metrics import mean_absolute_percentage_error
mae=mean_absolute_percentage_error(y_test,y_pred)
print(f'mae:{mae}')
RMSE2=np.sqrt(mean_squared_error(y_test,y_pred))
print(f'RMSE2:{RMSE2}')


# In[97]:


#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
model=rf.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[102]:


r2_score3=r2_score(y_test,y_pred)
print(f'r2_score3:{r2_score3}')




# In[106]:


from sklearn.metrics import mean_absolute_percentage_error
mae3=mean_absolute_percentage_error(y_test,y_pred)
print(f'mae3:{mae3}')



# In[105]:


rmse3=np.sqrt(mean_squared_error(y_test,y_pred))
print(f'rmse3:{rmse3}')


# In[ ]:


#Mean absolute percentage error is 7.3 percent and RMSE is 2734 which is less than the linear 
#regression and decisiontree model


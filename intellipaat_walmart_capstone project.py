#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


wf=pd.read_csv('Walmart DataSet.csv')
wf


# In[4]:


wf['Date']=pd.to_datetime(wf['Date'],format='mixed')
wf.sort_values(by=['Date'],ascending=True,inplace=True)
wf


# In[5]:


wf.index=wf['Date']


# In[6]:


wf.info()


# In[ ]:





# In[7]:


wf.isnull().sum()


# In[8]:


wf.info()


# In[9]:


wf.duplicated().sum()


# In[10]:


plt.plot(wf['Weekly_Sales'],wf['Unemployment'])
plt.show()


# In[ ]:





# In[11]:


wf.tail()


# In[ ]:





# In[ ]:





# In[12]:


gp=wf.groupby(['Store'])['Weekly_Sales'].sum().reset_index().sort_values(by=['Weekly_Sales'],ascending=True)
gp


# In[ ]:





# In[ ]:





# In[13]:


#a.store no 33 is suffering the most


# In[14]:


plt.figure(figsize=(10,12))
plt.plot(wf['Date'],wf['Holiday_Flag'])
plt.show()


# In[15]:


plt.figure(figsize=(10,12))
plt.plot(wf['Date'],wf['Temperature'])
plt.xlabel(wf['Date'])
plt.show()


# In[16]:


plt.figure(figsize=(10,12))
plt.plot(wf['Date'],wf['Fuel_Price'])
plt.xlabel(wf['Date'])
plt.show()


# In[17]:


plt.figure(figsize=(10,12))
plt.plot(wf['Date'],wf['CPI'])
plt.xlabel(wf['Date'])
plt.show()


# In[18]:


plt.figure(figsize=(10,12))
plt.plot(wf['Date'],wf['Unemployment'])
plt.xlabel(wf['Date'])
plt.show()


# In[ ]:





# In[19]:


#b. If the weekly sales show a seasonal trend, when and what could be the reason?
wf


# In[20]:


wf1=wf.drop(['Date'],axis=1)
wf1


# In[21]:


seansonal= wf1.iloc[:,:2]


# In[22]:


del seansonal['Store']


# In[23]:


seansonal


# In[24]:


seansonal.plot()


# In[25]:


#seasonal trend is 12 months ,every new year its showing the greatest weekly_sales and the reason behind is temprature and fuel_price
#when temp decreases and becomes minimum weely_sales is highest.


# In[26]:


#c.yes, when temperature is between 20 to 60 degree, weekly_sales is more as compare to 
#higher and lower of this temperature range
plt.figure(figsize=(10,12))
plt.plot(wf['Temperature'],wf['Weekly_Sales'])
plt.xlabel(wf['Temperature'])
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[27]:


#How is the Consumer Price index affecting the weekly sales of various stores?
plt.figure(figsize=(10,12))
plt.plot(wf['CPI'],wf['Weekly_Sales'])
plt.xlabel(wf['CPI'])
plt.show()


# In[28]:


#e. Top performing stores according to the historical data.
plt.figure(figsize=(10,12))
sns.barplot(x=wf['Store'],y=wf['Weekly_Sales'],color='green')
plt.xlabel(wf['Store'])
plt.show()
# store no 20,4 is having highest sales


# In[29]:


wf['year'] = pd.DatetimeIndex(wf['Date']).year
wf['month'] = pd.DatetimeIndex(wf['Date']).month

plt.figure(figsize=(15,7))
plt.scatter(wf[wf.year==2010]["month"],wf[wf.year==2010]["Weekly_Sales"])
plt.xlabel("Months")
plt.ylabel("Weekly Sales")
plt.title("Monthly view of sales in 2010")
plt.show()


# In[30]:


plt.figure(figsize=(15,7))
plt.scatter(wf[wf.year==2011]["month"],wf[wf.year==2011]["Weekly_Sales"])
plt.xlabel("Months")
plt.ylabel("Weekly Sales")
plt.title("Monthly view of sales in 2011")
plt.show()


# In[31]:


wf['year'] = pd.DatetimeIndex(wf['Date']).year
wf['month'] = pd.DatetimeIndex(wf['Date']).month
plt.figure(figsize=(15,7))
plt.scatter(wf[wf.year==2012]["month"],wf[wf.year==2012]["Weekly_Sales"])
plt.xlabel("Months")
plt.ylabel("Weekly Sales")
plt.title("Monthly view of sales in 2010")
plt.show()


# In[32]:


wf2=wf.copy()
wf2


# In[ ]:





# In[ ]:





# In[33]:


#f. The worst performing store, and how significant is the difference between the
#highest and lowest performing stores.
gp=wf.groupby(['Store'])['Weekly_Sales'].sum().reset_index()
gp.sort_values(by=['Weekly_Sales'],ascending=False)


# In[34]:


weekly_sales_diff= gp['Weekly_Sales'].max()-gp['Weekly_Sales'].min()
weekly_sales_diff


# In[35]:


#2. Use predictive modeling techniques to forecast the sales for each store for the next 12 weeks.


# In[36]:


seasonal=wf1.iloc[:,:2]
del seasonal['Store']
seasonal


# In[ ]:





# In[ ]:





# In[37]:


from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(seasonal.Weekly_Sales, period=12)  
plt.figure(figsize=(12,10))  
decomposition.plot()  
plt.show()


# In[38]:


from statsmodels.tsa.stattools import adfuller
result =adfuller(seasonal['Weekly_Sales'])
p_val =result[1]


# In[39]:


p_val


# In[40]:


if(p_val<0.05):
  print("we accept the null hypo - data is stationary")
else:
  print("we reject the null hypo - data is not stationary")


# In[41]:


#lets see for store 1
store1=wf1[(wf1['Store']==1)]
store1.tail(50)


# In[42]:


store1=store1.iloc[:,:2]
del store1['Store']


# In[43]:


store1


# In[44]:


store1_train=store1.loc['2010-01-10':'2012-01-06']
store1_test=store1.loc['2012-01-06':'2012-12-10']


# In[ ]:





# In[45]:


get_ipython().system('pip install pmdarima')


# In[46]:


from pmdarima import auto_arima
auto=auto_arima(store1['Weekly_Sales'])


# In[47]:


auto.summary()


# In[ ]:


# according to autoarima sarmiax(0,1,2) is the best model for the given dataset


# In[48]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[49]:


import warnings
warnings.filterwarnings('ignore')


# In[50]:


model=SARIMAX(store1_train,order=(0,1,2),seasonal_order=(0,1,2,12))


# In[51]:


model_fit=model.fit()


# In[52]:


store1['Predict']= model_fit.predict(start = len(store1_train),end=len(store1_train)+len(store1_test)-1)


# In[53]:


store1[['Weekly_Sales','Predict']].plot()
plt.figure(figsize=(12,10))
plt.show()


# In[54]:


forecast=model_fit.forecast(steps=12)


# In[ ]:





# In[55]:


store1.plot()
forecast.plot()


# In[56]:


#forcasting for store 2
store2=wf1[(wf1['Store']==2)]
store2.tail(50)


# In[58]:


store2=store2.iloc[:,:2]
del store2['Store']


# In[59]:


store2


# In[60]:


store2_train=store2.iloc[:110,:]
store2_test=store2.iloc[110:,:]


# In[61]:


store2_train


# In[62]:


from pmdarima import auto_arima
auto2=auto_arima(store2['Weekly_Sales'])


# In[63]:


auto.summary()


# In[64]:


model=SARIMAX(store2_train,order=(0,1,2),seasonal_order=(0,1,2,12))
model_fit=model.fit()
store2['Predict']= model_fit.predict(start = len(store2_train),end=len(store2_train)+len(store2_test)-1)
store2[['Weekly_Sales','Predict']].plot()
plt.figure(figsize=(12,10))
plt.show()


# In[65]:


forecast=model_fit.forecast(steps=12)
store2.plot()
forecast.plot()


# In[66]:


#lets forcast for store no.3
store3=wf1[(wf1['Store']==3)]
store3=store3.iloc[:,:2]
del store3['Store']
store3


# In[67]:


store3_train=store3.iloc[:110,:]
store3_test=store3.iloc[110:,:]


# In[68]:


model=SARIMAX(store3_train,order=(0,1,2),seasonal_order=(0,1,2,12))
model_fit=model.fit()
store3['Predict']= model_fit.predict(start = len(store3_train),end=len(store3_train)+len(store3_test)-1)
store3[['Weekly_Sales','Predict']].plot()
plt.figure(figsize=(12,10))
plt.show()


# In[69]:


forecast=model_fit.forecast(steps=12)
store3.plot()
forecast.plot()


# In[70]:


#lets check for store 4
store4=wf1[(wf1['Store']==4)]
store4=store4.iloc[:,:2]
del store4['Store']
store4


# In[71]:


store4_train=store4.iloc[:110,:]
store4_test=store4.iloc[110:,:]


# In[72]:


model=SARIMAX(store4_train,order=(0,1,2),seasonal_order=(0,1,2,12))
model_fit=model.fit()
store4['Predict']= model_fit.predict(start = len(store4_train),end=len(store4_train)+len(store4_test)-1)
store4[['Weekly_Sales','Predict']].plot()
plt.figure(figsize=(12,10))
plt.show()


# In[80]:


forecast=model_fit.forecast(steps=12)
store4.plot()
forecast.plot()


# In[75]:


#for store no 5
store5=wf1[(wf1['Store']==5)]
store5=store5.iloc[:,:2]
del store5['Store']
store5


# In[76]:


store5_train=store5.iloc[:110,:]
store5_test=store5.iloc[110:,:]


# In[77]:


model=SARIMAX(store5_train,order=(0,1,2),seasonal_order=(0,1,2,12))
model_fit=model.fit()
store5['Predict']= model_fit.predict(start = len(store5_train),end=len(store5_train)+len(store5_test)-1)
store5[['Weekly_Sales','Predict']].plot()
plt.figure(figsize=(12,10))
plt.show()


# In[79]:


forecast=model_fit.forecast(steps=12)
store5.plot()
forecast.plot()


# In[ ]:





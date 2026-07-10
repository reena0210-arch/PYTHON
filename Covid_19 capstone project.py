#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


cf= pd.read_csv('covid_19_clean_complete.csv')
cf


# In[4]:


cf.info()


# In[5]:


cf['Date']=pd.to_datetime(cf['Date'])
cf.info()


# In[6]:


cf['Country/Region'].nunique()


# In[7]:


cf_india=cf[cf['Country/Region']=='India'][['Date','Confirmed','Deaths', 'Recovered','Active' ]]


# In[8]:


cf_india


# In[121]:


cf_india_confirmed=cf_india[(cf_india['Confirmed']!=0)]
cf_india_confirmed


# In[122]:


#total confirmed cases in india
total_confirmed_inida=cf_india_confirmed['Confirmed'].sum(axis=0)
total_confirmed_inida


# In[123]:


#total recovered cases in india
cf_india_recovered=cf_india_confirmed['Recovered'].sum(axis=0)
cf_india_recovered


# In[124]:


#active cases in india
cf_india_active=cf_india_confirmed['Active'].sum(axis=0)
cf_india_active


# In[125]:


#total deaths
total_deaths_inindia=cf_india_confirmed['Deaths'].sum(axis=0)
total_deaths_inindia


# In[126]:


#recovery rate and deaths rate
recovery_rate_india=cf_india_recovered*100/total_confirmed_inida
deaths_rate_india=total_deaths_inindia*100/total_confirmed_inida
print(f'recovery rate in inida,{recovery_rate_india}')
print(f'deaths rate in inida,{deaths_rate_india}')


# In[127]:


#write code to visualize the impact
cf_india_confirmed.deaths= cf_india_confirmed.copy()


# In[128]:


cf_india_confirmed.deaths.index=cf_india_confirmed.deaths['Date']
cf_india_confirmed.deaths


# In[129]:


cf_india_confirmed.deaths.drop(columns=['Date','Active'],inplace=True)


# In[130]:


cf_india_confirmed.deaths.plot()
plt.show()


# In[131]:


#accroding to data we can see in india from feb 2020 to jul 2020 its not impacting much.


# In[132]:


cf['Country/Region'].unique()


# In[ ]:


#lets see for 'China' and 'US'


# In[53]:


cf_china=cf[cf['Country/Region']=='China'][['Date','Confirmed','Deaths', 'Recovered','Active' ]]


# In[54]:


cf_china.info()


# In[116]:


#confirmed cases
cf_china_confirmed=cf_china[(cf_china['Confirmed']!=0)]
cf_china_confirmed


# In[117]:


china_confirmed_cases=cf_china_confirmed['Confirmed'].sum(axis=0)
china_deaths=cf_china_confirmed['Deaths'].sum(axis=0)
china_recovered=cf_china_confirmed['Recovered'].sum(axis=0)
china_active=cf_china_confirmed['Active'].sum(axis=0)


# In[118]:


# recovery rate and deaths rate in china
recovery_rate_china=china_recovered*100/china_confirmed_cases
deaths_rate_china=china_deaths*100/china_confirmed_cases
print(f'recovery rate in china,{recovery_rate_china}')
print(f'deaths rate in china,{deaths_rate_china}')


# In[119]:


# by comparing both the coutry(inida and china), recovery rate is faster in china as compare to india 
# but death rate is greater in china as compare to india


# # trend of infection and recorvery in inida

# In[133]:


cf_india_confirmed1=cf_india_confirmed.copy()


# In[134]:


cf_india_confirmed1.drop(columns=['Deaths','Recovered','Active'],inplace=True)


# In[135]:


cf_india_confirmed1


# In[136]:


cf_india_confirmed1.plot()
plt.show()


# In[93]:


get_ipython().system('pip install prophet')


# In[87]:


import warnings
warnings.filterwarnings('ignore')


# In[95]:


from prophet import Prophet


# In[137]:


cf_india_train=cf_india_confirmed1.iloc[:120,:]
cf_india_test=cf_india_confirmed1.iloc[120:,:]


# In[138]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[139]:


decompose=seasonal_decompose(cf_india_train.Confirmed,model='additive',extrapolate_trend='freq', period=24)
decompose.plot().show()


# In[150]:


cf_india_prophet=cf_india_train
cf_india_prophet


# In[146]:


cf_india_prophet = cf_india_prophet.rename(columns={'Date':'ds','Confirmed': 'y'})
cf_india_prophet.info()


# In[147]:


model_prophet  = Prophet(interval_width=0.95)
model_prophet.fit(cf_india_prophet)


# In[148]:


model_prophet.component_modes


# In[151]:


future_dates= model_prophet.make_future_dataframe(periods=7)
future_dates.tail()


# In[165]:


forecast_prophet = model_prophet.predict(future_dates)
forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].round().tail(7)


# In[164]:


# plot the time series 
forecast_plot = model_prophet.plot(forecast_prophet)

# add a vertical line at the end of the training period
axes = forecast_plot.gca()
last_training_date = forecast_prophet['ds'].iloc[-7]
axes.axvline(x=last_training_date, color='red', linestyle='--', label='Training End')

# plot true test data for the period after the red line
plt.plot(cf_india_test['Date'], cf_india_test['Confirmed'],'ro', markersize=3, label='True Test Data')

# show the legend to distinguish between the lines
plt.legend()


# # forcasting for recovered cases in  india

# In[154]:


cf_india_recovered=cf_india[(cf_india['Recovered']!=0)]
cf_india_recovered


# In[155]:


cf_india_recovered.drop(columns=['Confirmed','Deaths','Active'], inplace=True)


# In[156]:


cf_india_recovered_prophet=cf_india_recovered.copy()
cf_india_recovered_prophet.info()


# In[157]:


cf_india_recovered_prophet=cf_india_recovered_prophet.rename(columns={'Date':'ds','Recovered':'y'})


# In[158]:


model2= Prophet(interval_width=0.95)
model2.fit(cf_india_recovered_prophet)


# In[160]:


future_date2=model2.make_future_dataframe(periods=7)
future_date2.tail(7)


# In[162]:


forecast_prophet = model2.predict(future_date2)
forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].round().tail()


# In[163]:


# plot the time series 
forecast_plot = model2.plot(forecast_prophet)

# add a vertical line at the end of the training period
axes = forecast_plot.gca()
last_training_date = forecast_prophet['ds'].iloc[-7]
axes.axvline(x=last_training_date, color='red', linestyle='--', label='Training End')

# plot true test data for the period after the red line
plt.plot(cf_india_test['Date'], cf_india_test['Confirmed'],'ro', markersize=3, label='True Test Data')

# show the legend to distinguish between the lines
plt.legend()


# In[168]:


model2.plot_components(forecast_prophet)


# In[175]:


#from fbprophet.diagnostics import cross_validation


# In[ ]:


#cf_india_cv=cross_validation(model2,initial='240 days',period='7 days',horizon='14 days')
#cf_india_cv.head()


# In[174]:


#from fbprophet.diagnostics import performance_metrics
#cf_inida_p = performance_metrics(cf_india_cv)
#cf_inida_p


# In[ ]:





# In[ ]:





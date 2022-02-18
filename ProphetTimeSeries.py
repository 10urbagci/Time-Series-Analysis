"""


@author: ASUS
"""

#Load necessery library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Load our Dataset
data = pd.read_csv("covid_19_data.csv")
#Transform data oputputs an index object containing date values
data['ObservationDate'] = pd.DatetimeIndex(data['ObservationDate'])
print(data.columns)
print(data.dtypes)

#Data preprocessing 
#We obtained the sum of the observation date data using groupby
df = data.groupby('ObservationDate').sum()
#We named the columns.
df.columns = ['sno','Confirmed_sum','Deaths_sum','Recovered_sum']
#We got the averages
df_mean = data.groupby('ObservationDate').mean()
df_mean.columns = ['s','Confirmed_mean','Deaths_mean','Recovered_mean']
#Join columns of another DataFrame It's like left join in SQL 
#We combined all the tables and assigned them to df
df = df.join(df_mean)
#We deleted s rows axis = 1 mean is delete rows 
df = df.drop(['s'],axis=1)


#Plotted the recovered_sum using matplotlib
df['Recovered_sum'].plot()



df['datetime'] = df.index


#Imported the datetime and Recovered_sum columns info df2
df2 = df[['datetime','Deaths_sum']]
#We named the columns ds and y Prophet only accepts ds and y names.
df2.columns = ['ds','y']
#iloc integer location The first : - take all rows. 
df2.index = list(range(df2.iloc[:,0].size))

#Load FB Prophet
from prophet import Prophet

print(df2.head())

#Initialize the Model
#create our model
#Fit our Model to our Data We fit the model by instantiating a new Prophet object
m = Prophet()
m.fit(df2)
#Shape of Datasets
print('Before shape:')
print(df2.shape)
#Create Future Dates of 365 Days
future = m.make_future_dataframe(periods=365)
future.tail()
#Shape after adding 365 days
print('After shape:')
print(future.shape)
#It gives first 5 columns after adding 365 days
print(future.head())

#Make Prediction with our Model
print('Prediction:')
prediction = m.predict(future)
print(prediction)
#It gives first 5 data
print(prediction.head())
#It gives last 5 data
print(prediction.tail())    

#The prediction method will assign each row an estimated value,which it calls yhat.
#ds -> dates 
# yhat_lower -> yhat_upper bound
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


#Visualize Components (Trends,Weekly)
#Plot the forecast 
fig1 = m.plot(forecast)
#plot trend,yearly,seasonality and weekly
fig2 = m.plot_components(forecast)

#Import Cross Validation
from fbprophet.diagnostics import cross_validation
#We cross-validate to evaluate forecast performance over a 365-day horizon
#starting with 60 days of training data, then making forecasts every 180 days.
cv = cross_validation(m,initial='60 days',period ='180 days',horizon='365 days')
print(cv.head())

#Cross validation performance metrices
from fbprophet.diagnostics import performance_metrics
df_pm = performance_metrics(cv)
print('Performance Metrices:')
print(df_pm)

#Cross validation performance metrices visualized
from fbprophet.plot import plot_cross_validation_metric
#Root Mean Square Error
plot_cross_validation_metric(cv,metric = 'rmse')


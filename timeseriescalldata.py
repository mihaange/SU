# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 22:02:35 2023

@author: MP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error


df = pd.read_csv("calldata.csv")
print(df.head())

averages = df.mean().round(1)
print(averages)

summary = df.describe().round(1)
print(summary)
#it includes the count, mean, standard deviation, minimum, 25th percentile, median (50th percentile), 75th percentile and maximum of each column.

def abandonment_rate(row):
    return (row['Incoming'] - row['Answered']) / row['Incoming'] * 100
df['Abandonment Rate'] = df.apply(abandonment_rate, axis=1)
print(df)

total_abandonment_rate = (df['Incoming'].sum() - df['Answered'].sum()) / df['Incoming'].sum() * 100
print("Total Abandonment Rate: ",total_abandonment_rate)

plt.plot(df['Incoming'], label='Incoming')
plt.plot(df['Answered'], label='Answered')
plt.plot(df['Abandoned'], label='Abandoned')
plt.xlabel('Time')
plt.ylabel('Number of calls')
plt.legend()
plt.show()

labels = ['Answered', 'Abandoned']
sizes = [df['Answered'].sum(), df['Abandoned'].sum()]
colors = ['lightgreen', 'red']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
plt.axis('equal')
plt.title('Proportion of Answered and Abandoned Calls')
plt.show()

std = df.std()
plt.plot(averages, label='Averages')
plt.plot(std, label='Standard Deviation')
plt.xlabel('Columns')
plt.ylabel('Values')
plt.title('Averages and Standard Deviation')
plt.legend()
plt.show()

df.reset_index(inplace=True)
df.rename(columns={'index':'Time'}, inplace=True)
df.set_index('Time', inplace=True)
train_data = df[:int(len(df)*0.8)]
test_data = df[int(len(df)*0.8):]

mae_values = {}
rmse_values = {}

columns = ['Incoming', 'Answered', 'Abandoned']
for column in columns:
    model = ARIMA(train_data[column], order=(2, 1, 2))
    model_fit = model.fit(disp=False)
    predictions, _, _ = model_fit.forecast(steps=len(test_data))
    
    mse = mean_squared_error(test_data[column], predictions)
    mae = mean_absolute_error(test_data[column], predictions)
    rmse = np.sqrt(mean_squared_error(test_data[column], predictions))
    
    plt.plot(train_data[column], label='Train Data')
    plt.plot(test_data[column], label='Test Data')
    plt.plot(predictions, label='Predictions', color='red')
    plt.xlabel('Time')
    plt.ylabel(column)
    plt.legend()
    plt.show()
    print("Mean Squared Error for column {} is : {}".format(column,mse))
    
    mae_values[column] = mae
    rmse_values[column] = rmse
    print("Mean Absolute Error for column {} is : {}".format(column,mae))
    print("Root Mean Squared Error for column {} is : {}".format(column,rmse))

plt.bar(mae_values.keys(), mae_values.values())
plt.xlabel('Columns')
plt.ylabel('MAE')
plt.title('Mean Absolute Error')
plt.show()
    
plt.bar(rmse_values.keys(), rmse_values.values())
plt.xlabel('Columns')
plt.ylabel('RMSE')
plt.title('Root Mean Squared Error')
plt.show()

columns = ['Incoming', 'Answered', 'Abandoned']
for column in columns:
    svr = SVR(kernel='linear')
    svr.fit(train_data.index.values.reshape(-1,1), train_data[column].values)
    predictions = svr.predict(test_data.index.values.reshape(-1,1))
    mse = mean_squared_error(test_data[column], predictions)
    mae = mean_absolute_error(test_data[column], predictions)
    rmse = np.sqrt(mean_squared_error(test_data[column], predictions))
    print("SVR linear:")
    plt.plot(train_data[column], label='Train Data')
    plt.plot(test_data[column], label='Test Data')
    plt.plot(predictions, label='Predictions', color='red')
    plt.xlabel('Time')
    plt.ylabel(column)
    plt.legend()
    plt.show()
    
    print("Mean Squared Error for column {} is : {}".format(column,mse))
    
    mae_values[column] = mae
    rmse_values[column] = rmse
    print("Mean Absolute Error for column {} is : {}".format(column,mae))
    print("Root Mean Squared Error for column {} is : {}".format(column,rmse))


plt.bar(mae_values.keys(), mae_values.values())
plt.xlabel('Columns')
plt.ylabel('Mean Absolute Error')
plt.title('MAE')
plt.show()
plt.bar(rmse_values.keys(), rmse_values.values())
plt.xlabel('Columns')
plt.ylabel('Root Mean Squared Error')
plt.title('RMSE')
plt.show()

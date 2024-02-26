# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('Historical_Data.csv')
print(df.shape)
print(df.columns)
print(df.dtypes)

# Convert Date to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
df = df.set_index('Date')
print(df.head())

# Group by Date and sum Sold_Units
df = df.groupby('Date')['Sold_Units'].sum()

# Set the frequency of the date index to daily
df = df.asfreq('D')
print(df.head())

# Plot the time series
plt.figure(figsize=(12,6))
plt.plot(df)
plt.title('Daily Sales Time Series')
plt.xlabel('Date')
plt.ylabel('Sold_Units')
plt.show()

# Split into train and test sets
train = df[:-30]
test = df[-30:]

# Choose a model, e.g. SARIMA
model = sm.tsa.statespace.SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,7))
model_fit = model.fit()

# Evaluate the model on the test set
y_pred = model_fit.forecast(30)
y_pred.index = test.index  # Align the index of the predictions with the test set

# Recalculate the metrics
mae = np.mean(np.abs(test - y_pred))
rmse = np.sqrt(np.mean((test - y_pred)**2))
mape = np.mean(np.abs((test - y_pred) / test)) * 100

print('MAE: {:.2f}'.format(mae))
print('RMSE: {:.2f}'.format(rmse))
print('MAPE: {:.2f}%'.format(mape))

# Plot the actual and predicted values
plt.figure(figsize=(12,6))
plt.plot(test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Daily Sales Time Series - Test Set')
plt.xlabel('Date')
plt.ylabel('Sold_Units')
plt.legend()
plt.show()

# Forecast the future demand
y_forecast = model_fit.forecast(30)
y_forecast.index = pd.date_range(start=test.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')  # Create a date range for the forecast

plt.figure(figsize=(12,6))
plt.plot(y_forecast)
plt.title('Daily Sales Time Series - Forecast')
plt.xlabel('Date')
plt.ylabel('Sold_Units')
plt.show()

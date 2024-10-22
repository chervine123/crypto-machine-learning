from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#############
# ADA plots #
#############

#reading & prepping data for Prophet modelling
df = pd.read_csv('ADA-USD_historical_data.csv')
df = df[['Date', 'Adj Close']]
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df = df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})

#Splitting data into 80% training and 20% testing datasets
train_size = int(len(df) * 0.8)  # 80% for training
train_df = df[:train_size]
test_df = df[train_size:]

#Fit the Prophet Model on the training set
model = Prophet()
model.fit(train_df)

#Make predictions for the test period
future = model.make_future_dataframe(periods=len(test_df))
forecast = model.predict(future)

# Extract the predicted values for the test set
forecast_test = forecast[-len(test_df):]  # Get predictions for the test period
predicted_test = forecast_test['yhat'].values
actual_test = test_df['y'].values

#Calculate RMSE
error = actual_test - predicted_test
squared_error = np.square(error)
mean_squared_error = np.mean(squared_error)
rmse = np.sqrt(mean_squared_error)
print(f'ADA RMSE: {rmse}')

#Plotting Actual vs Predicted values with confidence intervals
plt.figure(figsize=(10, 6))
# Plot actual data (blue line)
plt.plot(train_df['ds'], train_df['y'], label='Training Data', color='blue')
plt.plot(test_df['ds'], test_df['y'], label='Actual', color='blue', linestyle='--')
# Plot predicted data (red line)
plt.plot(forecast_test['ds'], predicted_test, label='Predicted Prices', color='red', linestyle='--')
# Plot the confidence interval (fan chart)
plt.fill_between(forecast_test['ds'], 
                 forecast_test['yhat_lower'], 
                 forecast_test['yhat_upper'], 
                 color='grey', alpha=0.3, label='Confidence Interval')
# Plot
plt.title('ADA - Actual vs Predicted Adj Close with Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()




#############
# BNB plots #
#############

#reading & prepping data for Prophet modelling
df = pd.read_csv('BNB-USD_historical_data.csv')
df = df[['Date', 'Adj Close']]
df['Date'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
print(df)

#Splitting data into 80% training and 20% testing datasets
train_size = int(len(df) * 0.8)  # 80% for training
train_df = df[:train_size]
test_df = df[train_size:]

#Fit the Prophet Model on the training set
model = Prophet()
model.fit(train_df)

#Make predictions for the test period
future = model.make_future_dataframe(periods=len(test_df))
forecast = model.predict(future)

# Extract the predicted values for the test set
forecast_test = forecast[-len(test_df):]  # Get predictions for the test period
predicted_test = forecast_test['yhat'].values
actual_test = test_df['y'].values

#Calculate RMSE
error = actual_test - predicted_test
squared_error = np.square(error)
mean_squared_error = np.mean(squared_error)
rmse = np.sqrt(mean_squared_error)
print(f'BNB RMSE: {rmse}')

#Plotting Actual vs Predicted values with confidence intervals
plt.figure(figsize=(10, 6))
# Plot actual data (blue line)
plt.plot(train_df['ds'], train_df['y'], label='Training Data', color='blue')
plt.plot(test_df['ds'], test_df['y'], label='Actual', color='blue', linestyle='--')
# Plot predicted data (red line)
plt.plot(forecast_test['ds'], predicted_test, label='Predicted Prices', color='red', linestyle='--')
# Plot the confidence interval (fan chart)
plt.fill_between(forecast_test['ds'], 
                 forecast_test['yhat_lower'], 
                 forecast_test['yhat_upper'], 
                 color='grey', alpha=0.3, label='Confidence Interval')
# Plot
plt.title('BNB - Actual vs Predicted Adj Close with Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()


#############
# BTC plots #
#############

#reading & prepping data for Prophet modelling
df = pd.read_csv('BTC-USD_historical_data.csv')
df = df[['Date', 'Adj Close']]
df['Date'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
print(df)

#Splitting data into 80% training and 20% testing datasets
train_size = int(len(df) * 0.8)  # 80% for training
train_df = df[:train_size]
test_df = df[train_size:]

#Fit the Prophet Model on the training set
model = Prophet()
model.fit(train_df)

#Make predictions for the test period
future = model.make_future_dataframe(periods=len(test_df))
forecast = model.predict(future)

# Extract the predicted values for the test set
forecast_test = forecast[-len(test_df):]  # Get predictions for the test period
predicted_test = forecast_test['yhat'].values
actual_test = test_df['y'].values

#Calculate RMSE
error = actual_test - predicted_test
squared_error = np.square(error)
mean_squared_error = np.mean(squared_error)
rmse = np.sqrt(mean_squared_error)
print(f'BTC RMSE: {rmse}')

#Plotting Actual vs Predicted values with confidence intervals
plt.figure(figsize=(10, 6))
# Plot actual data (blue line)
plt.plot(train_df['ds'], train_df['y'], label='Training Data', color='blue')
plt.plot(test_df['ds'], test_df['y'], label='Actual', color='blue', linestyle='--')
# Plot predicted data (red line)
plt.plot(forecast_test['ds'], predicted_test, label='Predicted Prices', color='red', linestyle='--')
# Plot the confidence interval (fan chart)
plt.fill_between(forecast_test['ds'], 
                 forecast_test['yhat_lower'], 
                 forecast_test['yhat_upper'], 
                 color='grey', alpha=0.3, label='Confidence Interval')
# Plot
plt.title('BTC - Actual vs Predicted Adj Close with Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()




#############
# ETH plots #
#############

#reading & prepping data for Prophet modelling
df = pd.read_csv('ETH-USD_historical_data.csv')
df = df[['Date', 'Adj Close']]
df['Date'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
print(df)

#Splitting data into 80% training and 20% testing datasets
train_size = int(len(df) * 0.8)  # 80% for training
train_df = df[:train_size]
test_df = df[train_size:]

#Fit the Prophet Model on the training set
model = Prophet()
model.fit(train_df)

#Make predictions for the test period
future = model.make_future_dataframe(periods=len(test_df))
forecast = model.predict(future)

# Extract the predicted values for the test set
forecast_test = forecast[-len(test_df):]  # Get predictions for the test period
predicted_test = forecast_test['yhat'].values
actual_test = test_df['y'].values

#Calculate RMSE
error = actual_test - predicted_test
squared_error = np.square(error)
mean_squared_error = np.mean(squared_error)
rmse = np.sqrt(mean_squared_error)
print(f'ETH RMSE: {rmse}')

#Plotting Actual vs Predicted values with confidence intervals
plt.figure(figsize=(10, 6))
# Plot actual data (blue line)
plt.plot(train_df['ds'], train_df['y'], label='Training Data', color='blue')
plt.plot(test_df['ds'], test_df['y'], label='Actual', color='blue', linestyle='--')
# Plot predicted data (red line)
plt.plot(forecast_test['ds'], predicted_test, label='Predicted Prices', color='red', linestyle='--')
# Plot the confidence interval (fan chart)
plt.fill_between(forecast_test['ds'], 
                 forecast_test['yhat_lower'], 
                 forecast_test['yhat_upper'], 
                 color='grey', alpha=0.3, label='Confidence Interval')
# Plot
plt.title('ETH - Actual vs Predicted Adj Close with Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()



#############
# BNB plots #
#############

#reading & prepping data for Prophet modelling
df = pd.read_csv('SOL-USD_historical_data.csv')
df = df[['Date', 'Adj Close']]
df['Date'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
print(df)

#Splitting data into 80% training and 20% testing datasets
train_size = int(len(df) * 0.8)  # 80% for training
train_df = df[:train_size]
test_df = df[train_size:]

#Fit the Prophet Model on the training set
model = Prophet()
model.fit(train_df)

#Make predictions for the test period
future = model.make_future_dataframe(periods=len(test_df))
forecast = model.predict(future)

# Extract the predicted values for the test set
forecast_test = forecast[-len(test_df):]  # Get predictions for the test period
predicted_test = forecast_test['yhat'].values
actual_test = test_df['y'].values

#Calculate RMSE
error = actual_test - predicted_test
squared_error = np.square(error)
mean_squared_error = np.mean(squared_error)
rmse = np.sqrt(mean_squared_error)
print(f'SOL RMSE: {rmse}')

#Plotting Actual vs Predicted values with confidence intervals
plt.figure(figsize=(10, 6))
# Plot actual data (blue line)
plt.plot(train_df['ds'], train_df['y'], label='Training Data', color='blue')
plt.plot(test_df['ds'], test_df['y'], label='Actual', color='blue', linestyle='--')
# Plot predicted data (red line)
plt.plot(forecast_test['ds'], predicted_test, label='Predicted Prices', color='red', linestyle='--')
# Plot the confidence interval (fan chart)
plt.fill_between(forecast_test['ds'], 
                 forecast_test['yhat_lower'], 
                 forecast_test['yhat_upper'], 
                 color='grey', alpha=0.3, label='Confidence Interval')
# Plot
plt.title('SOL - Actual vs Predicted Adj Close with Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()



#############
# XRP plots #
#############

#reading & prepping data for Prophet modelling
df = pd.read_csv('XRP-USD_historical_data.csv')
df = df[['Date', 'Adj Close']]
df['Date'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
print(df)

#Splitting data into 80% training and 20% testing datasets
train_size = int(len(df) * 0.8)  # 80% for training
train_df = df[:train_size]
test_df = df[train_size:]

#Fit the Prophet Model on the training set
model = Prophet()
model.fit(train_df)

#Make predictions for the test period
future = model.make_future_dataframe(periods=len(test_df))
forecast = model.predict(future)

# Extract the predicted values for the test set
forecast_test = forecast[-len(test_df):]  # Get predictions for the test period
predicted_test = forecast_test['yhat'].values
actual_test = test_df['y'].values

#Calculate RMSE
error = actual_test - predicted_test
squared_error = np.square(error)
mean_squared_error = np.mean(squared_error)
rmse = np.sqrt(mean_squared_error)
print(f'XRP RMSE: {rmse}')

#Plotting Actual vs Predicted values with confidence intervals
plt.figure(figsize=(10, 6))
# Plot actual data (blue line)
plt.plot(train_df['ds'], train_df['y'], label='Training Data', color='blue')
plt.plot(test_df['ds'], test_df['y'], label='Actual', color='blue', linestyle='--')
# Plot predicted data (red line)
plt.plot(forecast_test['ds'], predicted_test, label='Predicted', color='red', linestyle='--')
# Plot the confidence interval (fan chart)
plt.fill_between(forecast_test['ds'], 
                 forecast_test['yhat_lower'], 
                 forecast_test['yhat_upper'], 
                 color='grey', alpha=0.3, label='Confidence Interval')
# Plot
plt.title('XRP - Actual vs Predicted Adj Close with Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()

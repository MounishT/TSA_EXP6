# Ex.No: 6   HOLT WINTERS METHOD

## AIM:
To implement the Holt Winters Method Model using Python

### ALGORITHM:
1. Load and resample the Rainfall data to monthly frequency, selecting the 'date' column.
2. Scale the data using Minmaxscaler then split into training (80%) and testing (20%) sets.
3. Fit an additive Holt-Winters model to the training data and forecast on the test data.
4. Evaluate model performance using MAE and RMSE, and plot the train, test, and prediction results.
5. Train a final multiplicative Holt-Winters model on the full dataset and forecast future Rainfall.
### PROGRAM:
```
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Read the CSV without setting the index initially
data = pd.read_csv('/content/rainfall.csv')  

# Print the column names to see the actual header
print(data.columns)  

# Assuming the date column is named 'date' or 'DATE', update the code:
data = pd.read_csv('/content/rainfall.csv', index_col='date', parse_dates=True)  # Or use 'DATE' if that's the column name

# If there is an extra header row, skip it:
# data = pd.read_csv('/content/rainfall.csv', skiprows=1, index_col='date', parse_dates=True)
data = data.resample('MS').mean() 

# Select the 'PRICE' column for analysis
data = data['rainfall'] # selecting the rainfall column here

# Scaling the Data using MinMaxScaler 
scaler = MinMaxScaler()
data_scaled = pd.Series(scaler.fit_transform(data.values.reshape(-1, 1)).flatten(), index=data.index)

# Split into training and testing sets (80% train, 20% test)
train_data = data_scaled[:int(len(data_scaled) * 0.8)]
test_data = data_scaled[int(len(data_scaled) * 0.8):]

# Check if you have enough data for seasonality
if len(train_data) < 2 * 12:
    print("Warning: Not enough data for reliable seasonal modeling. Consider using a simpler model or gathering more data.")
    # Either proceed with caution or switch to a simpler model like SimpleExpSmoothing or Holt
    fitted_model_add = ExponentialSmoothing(train_data, trend='add').fit()  # Removing seasonal component
else:
    fitted_model_add = ExponentialSmoothing(
        train_data, trend='add', seasonal='add', seasonal_periods=12
    ).fit()

# Forecast and evaluate
test_predictions_add = fitted_model_add.forecast(len(test_data))

# Evaluate performance
print("MAE :", mean_absolute_error(test_data, test_predictions_add))
print("RMSE :", mean_squared_error(test_data, test_predictions_add, squared=False))

# Plot predictions
plt.figure(figsize=(12, 8))
plt.plot(train_data, label='TRAIN', color='black')
plt.plot(test_data, label='TEST', color='green')
plt.plot(test_predictions_add, label='PREDICTION', color='red')
plt.title('Train, Test, and Additive Holt-Winters Predictions')
plt.legend(loc='best')
plt.show()

# Similar check for the final model:
if len(data) < 2 * 12:
    print("Warning: Not enough data for reliable seasonal modeling in the final model.")
    final_model = ExponentialSmoothing(data, trend='mul').fit()  # Removing seasonal component
else:
    final_model = ExponentialSmoothing(data, trend='mul', seasonal='mul', seasonal_periods=12).fit()

# Forecast future values
forecast_predictions = final_model.forecast(steps=12)

data.plot(figsize=(12, 8), legend=True, label='Current rainfall')
forecast_predictions.plot(legend=True, label='Forecasted rainfall') # Calling .plot() on forecast_predictions
plt.xlabel('percentage')
plt.ylabel('date')
plt.title('Rainfall')
plt.show()
```
## OUTPUT:
#### TEST_PREDICTION
![image](https://github.com/user-attachments/assets/1804dcc2-3477-450a-a342-38ddaa34eb8f)

### FINAL_PREDICTION

![image](https://github.com/user-attachments/assets/e70de3f1-6473-4f36-86b5-1a62645dd241)

## RESULT:
Thus the program run successfully based on the Holt Winters Method model.

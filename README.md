# Ex.No: 6               HOLT WINTERS METHOD
### Date: 18-05-2026



### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose


# Load dataset
data = pd.read_csv("dataset.csv")

# Use first 120 rows to avoid overflow issue
data = data.head(120)

# Create monthly dates
data['Month'] = pd.date_range(
    start='2010-01-01',
    periods=len(data),
    freq='MS'
)

# Set index
data.set_index('Month', inplace=True)

# Select column for forecasting
ts_data = data['temp_mean(c)']


# Scale the data
scaler = MinMaxScaler()

scaled_data = pd.Series(
    scaler.fit_transform(ts_data.values.reshape(-1,1)).flatten(),
    index=ts_data.index
)

print("Scaled_data plot:")
scaled_data.plot(figsize=(8,4))
plt.show()


# Decompose
print("Decomposed plot:")

decomposition = seasonal_decompose(
    scaled_data,
    model='additive',
    period=12
)

decomposition.plot()
plt.show()


# Multiplicative seasonality requires positive values
scaled_data = scaled_data + 1


# Train-test split
train_data = scaled_data[:int(len(scaled_data)*0.8)]
test_data = scaled_data[int(len(scaled_data)*0.8):]


# Build model
model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()


# Test predictions
test_predictions_add = model_add.forecast(steps=len(test_data))

print("Test prediction:")

ax = train_data.plot(figsize=(8,4))
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)

ax.legend(["train_data", "test_predictions_add", "test_data"])
ax.set_title("Visual evaluation")

plt.show()


# Performance metrics
print("Model performance metrics:")

rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))

print("RMSE:")
print(rmse)

print("\nStandard deviation and mean:")
print((np.sqrt(scaled_data.var()), scaled_data.mean()))


# Final model
final_model = ExponentialSmoothing(
    scaled_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()


# Future prediction
final_predictions = final_model.forecast(steps=12)

print("\nFinal predictions:")

ax = scaled_data.plot(figsize=(8,4))
final_predictions.plot(ax=ax)

ax.legend(["data_monthly", "final_predictions"])
ax.set_title("Prediction")

plt.show()
```
### OUTPUT:

<img width="761" height="398" alt="image" src="https://github.com/user-attachments/assets/ac9259b0-ee6e-4938-b443-9c8b7b2c6465" />


<img width="728" height="486" alt="image" src="https://github.com/user-attachments/assets/fe0f570c-65d6-45dd-b911-38cb6d4a8123" />

TEST_PREDICTION

<img width="742" height="480" alt="image" src="https://github.com/user-attachments/assets/5388091a-c2c9-4f69-8597-fe99a8fbd4aa" />



FINAL_PREDICTION

<img width="746" height="422" alt="image" src="https://github.com/user-attachments/assets/b9c3cc35-fa35-4692-b2f2-ac906924a30a" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.

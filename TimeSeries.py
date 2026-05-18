# Name: Kamalakar Reddy
# UMBC ID: CJ88254
# Course: CMSC 462

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
rain = pd.read_csv("rain_hourly_clean.csv")

# Replace invalid sentinel value with NaN
rain['precip_mm'] = rain['precip_mm'].replace(999.99, np.nan)

# Drop those rows
rain = rain.dropna(subset=['precip_mm'])

# Q22

# Parse datetime and sort
rain['datetime'] = pd.to_datetime(rain['datetime'])
rain = rain.sort_values('datetime')
rain = rain.set_index('datetime')

# # Check spacing
# full_range = pd.date_range(rain.index.min(), rain.index.max(), freq='h')
# missing_hours = full_range.difference(rain.index)

# print("Total observations:", len(rain))
# print("Expected hourly timestamps:", len(full_range))
# print("Missing hours:", len(missing_hours))

# # Show first few missing timestamps
# print(missing_hours[:10])

# # Q23

# # F6
# plt.figure(figsize=(14,5))
# plt.plot(rain.index, rain['precip_mm'])
# plt.title('Hourly Precipitation Over Time')
# plt.xlabel('Date')
# plt.ylabel('Precipitation (mm)')
# plt.show()

# # Q24

# # F7

# Aggregate to daily totals
rain_daily = rain['precip_mm'].resample('D').sum()

plt.figure(figsize=(14,5))
plt.plot(rain_daily.index, rain_daily)
plt.title('Daily Total Precipitation')
plt.xlabel('Date')
plt.ylabel('Daily Precipitation (mm)')
plt.show()

# Use daily series for decomposition
rain_daily = rain_daily.fillna(0)

# # Decompose using weekly seasonality
# result = seasonal_decompose(rain_daily, model='additive', period=7)

# result.plot()
# plt.show()

# # ACF Plot
# plt.figure(figsize=(10,5))
# plot_acf(rain_daily, lags=30)
# plt.title("ACF of Daily Precipitation")
# plt.show()

# # PACF Plot
# plt.figure(figsize=(10,5))
# plot_pacf(rain_daily, lags=30)
# plt.title("PACF of Daily Precipitation")
# plt.show()

# Log transform to reduce skewness
rain_log = np.log1p(rain_daily)

# Fit AR model
model = AutoReg(rain_log, lags=1).fit()

print(model.summary())
print("AR Coefficients:")
print(model.params)
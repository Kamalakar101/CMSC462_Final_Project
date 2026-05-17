# Name: Kamalakar Reddy
# UMBC ID: CJ88254
# Course: CMSC 462

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



df1 = pd.read_csv("clean_daily_weather.csv")
df2 = pd.read_csv("rain_hourly_clean.csv")


# # Q4 Checking for duplicate date values
# print("Duplicate date values:", df1["DATE"].duplicated().any())

# # F1 - F3

# # Variables to plot
# variables = ["TMAX", "PRCP", "AWND"]

# # Creates one figure per variable
# for var in variables:
#     data = df1[var].dropna()

#     plt.figure(figsize=(8,5))
#     plt.hist(data, bins="fd")
#     plt.title(f"Histogram of {var}")
#     plt.xlabel(var)
#     plt.ylabel("Frequency")
#     plt.show()


# # F4

# numeric_df = df1.select_dtypes(include=['number'])

# # Compute correlation matrix
# corr_matrix = numeric_df.corr()

# # Display correlation matrix
# print(corr_matrix)

# # Plot heatmap
# plt.figure(figsize=(8,6))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Correlation Matrix Heatmap")
# plt.show()

# # F5
# plt.figure(figsize=(8,5))
# plt.scatter(df1["AWND"], df1["TMAX"])

# plt.title("AWND vs TMAX")
# plt.xlabel("Average Wind Speed (AWND)")
# plt.ylabel("Maximum Temperature (TMAX)")

# plt.show()

## Q10 - Q12

# 1. Temperature range
df1["temp_range"] = df1["TMAX"] - df1["TMIN"]

# 2. Rain day indicator
df1["rain_day"] = (df1["PRCP"] > 0).astype(int)

# 3. Wind gust ratio
df1["wind_gust_ratio"] = df1["WSF5"] / df1["AWND"]

# summary_stats = df1[["temp_range", "rain_day", "wind_gust_ratio"]].agg(["mean", "std", "min", "max"]).round(2)

# summary_stats.to_csv("summary_statistics.csv")

# df1["hot_day"] = (df1["TMAX"] >= 90).astype(int)

# Q13

# # Extract month from DATE column
# df1["DATE"] = pd.to_datetime(df1["DATE"])
# df1["Month"] = df1["DATE"].dt.month

# # January and July temperatures
# jan = df1[df1["Month"] == 1]["TAVG"].dropna()
# jul = df1[df1["Month"] == 7]["TAVG"].dropna()

# # Two-sample t-test
# t_stat, p_value = ttest_ind(jan, jul, equal_var=False)

# # Estimated difference in means
# mean_diff = jul.mean() - jan.mean()

# # 95% confidence interval
# se = np.sqrt(jan.var(ddof=1)/len(jan) + jul.var(ddof=1)/len(jul))

# df = len(jan) + len(jul) - 2
# critical = stats.t.ppf(0.975, df)

# ci_lower = mean_diff - critical * se
# ci_upper = mean_diff + critical * se

# # Print results
# print("January Mean:", jan.mean())
# print("July Mean:", jul.mean())
# print("Difference (July - January):", mean_diff)
# print("95% CI:", (ci_lower, ci_upper))
# print("p-value:", p_value)

# # Q14

# # Split data
# rain = df1[df1["rain_day"] == 1]["wind_gust_ratio"].dropna()
# no_rain = df1[df1["rain_day"] == 0]["wind_gust_ratio"].dropna()

# # Means
# mean_rain = rain.mean()
# mean_no_rain = no_rain.mean()

# diff = mean_rain - mean_no_rain

# # Welch two-sample t-test
# t_stat, p_value = ttest_ind(rain, no_rain, equal_var=False)

# # 95% CI calculation
# n1, n2 = len(rain), len(no_rain)
# var1, var2 = rain.var(ddof=1), no_rain.var(ddof=1)

# se = np.sqrt(var1/n1 + var2/n2)

# dfree = (var1/n1 + var2/n2)**2 / (
#     (var1**2)/(n1**2*(n1-1)) + (var2**2)/(n2**2*(n2-1))
# )

# t_crit = stats.t.ppf(0.975, dfree)

# ci_lower = diff - t_crit * se
# ci_upper = diff + t_crit * se

# # Output results
# print("Rain mean:", mean_rain)
# print("No rain mean:", mean_no_rain)
# print("Difference (rain - no rain):", diff)
# print("95% CI:", (ci_lower, ci_upper))
# print("p-value:", p_value)

# # Q15

# # Correlation between temp_range and AWND

# corr = df1["temp_range"].corr(df1["AWND"])

# # Drop missing values for CI calculation
# temp_awnd = df1[["temp_range", "AWND"]].dropna()

# n = len(temp_awnd)

# # Fisher z-transformation for 95% CI
# z = np.arctanh(corr)
# se = 1 / np.sqrt(n - 3)

# z_lower = z - 1.96 * se
# z_upper = z + 1.96 * se

# ci_lower = np.tanh(z_lower)
# ci_upper = np.tanh(z_upper)

# # Output
# print("Correlation:", corr)
# print("95% CI:", (ci_lower, ci_upper))

# Q16

# Predictors
predictors = ["TMIN", "AWND", "rain_day", "wind_gust_ratio"]

# Remove missing values
data = df1[predictors + ["TMAX"]].dropna()

X = data[predictors]
y = data["TMAX"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print coefficients
print("Intercept:", model.intercept_)

for feature, coef in zip(predictors, model.coef_):
    print(feature, ":", coef)

# Q17

print("\nRMSE:", rmse)
print("MAE:", mae)
print("R^2:", r2)
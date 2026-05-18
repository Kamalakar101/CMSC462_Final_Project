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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, roc_curve, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster



df1 = pd.read_csv("clean_daily_weather.csv")


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

# # Q16

# # Predictors
# predictors = ["TMIN", "AWND", "rain_day", "wind_gust_ratio"]

# # Remove missing values
# data = df1[predictors + ["TMAX"]].dropna()

# X = data[predictors]
# y = data["TMAX"]

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Fit model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Predictions
# y_pred = model.predict(X_test)

# # Metrics
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# # Print coefficients
# print("Intercept:", model.intercept_)

# for feature, coef in zip(predictors, model.coef_):
#     print(feature, ":", coef)

# # Q17

# print("\nRMSE:", rmse)
# print("MAE:", mae)
# print("R^2:", r2)

# # Q19

# # Predictors
# predictors = ["TAVG", "temp_range", "AWND", "wind_gust_ratio"]

# # Remove missing values
# data = df1[predictors + ["rain_day"]].dropna()

# X = data[predictors]
# y = data["rain_day"]

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42
# )

# # Fit logistic regression
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# # Predictions
# y_pred = model.predict(X_test)

# # Predicted probabilities
# y_prob = model.predict_proba(X_test)[:, 1]

# # Q20

# cm = confusion_matrix(y_test, y_pred)

# print("Confusion Matrix:")
# print(cm)

# # Accuracy
# accuracy = accuracy_score(y_test, y_pred)

# print("\nAccuracy:", accuracy)

# # ROC values
# fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# # AUC
# auc = roc_auc_score(y_test, y_prob)

# print("AUC:", auc)

# # Plot ROC curve
# plt.figure(figsize=(6,6))
# plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
# plt.plot([0,1], [0,1], linestyle="--")

# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()

# plt.show()

# for feature, coef in zip(predictors, model.coef_[0]):
#     print(feature, ":", coef)

# -------------------------
# Q28: Feature Selection + Scaling
# -------------------------

# Select features (must include engineered feature)
cluster_features = ["TAVG", "AWND", "temp_range", "wind_gust_ratio"]

# Drop missing values
cluster_data = df1[cluster_features].dropna()

# Standardize using z-scores
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_data)


# Q29: K-Means Clustering

# Elbow method to choose k
inertia = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.show()

# Choose k (based on elbow, usually 3 or 4 for weather patterns)
k = 2

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters_km = kmeans.fit_predict(X_scaled)

cluster_data["kmeans_cluster"] = clusters_km


# Hierarchical Clustering

linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(10,5))
dendrogram(linked, truncate_mode='level', p=5)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

# Cut dendrogram into same k clusters
clusters_hc = fcluster(linked, k, criterion='maxclust')
cluster_data["hier_cluster"] = clusters_hc


# F8: Scatter Plot of Clusters

plt.figure(figsize=(7,5))
plt.scatter(cluster_data["TAVG"], cluster_data["AWND"], 
            c=clusters_km, cmap="viridis", s=20)

plt.title("K-Means Clusters (TAVG vs AWND)")
plt.xlabel("Average Temperature (TAVG)")
plt.ylabel("Wind Speed (AWND)")
plt.show()

# Q30: Cluster Interpretation Support

print("\nCluster Means (K-Means):")
print(cluster_data.groupby("kmeans_cluster")[cluster_features].mean())

print("\nCluster Sizes:")
print(cluster_data["kmeans_cluster"].value_counts().sort_index())
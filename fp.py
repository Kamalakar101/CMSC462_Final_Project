# Name: Kamalakar Reddy
# UMBC ID: CJ88254
# Course: CMSC 462

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df1 = pd.read_csv("clean_daily_weather.csv")
df2 = pd.read_csv("rain_hourly_clean.csv")


# Q4 Checking for duplicate date values
print("Duplicate date values:", df1["DATE"].duplicated().any())

# F1 - F3

# Variables to plot
variables = ["TMAX", "PRCP", "AWND"]

# Creates one figure per variable
for var in variables:
    data = df1[var].dropna()

    plt.figure(figsize=(8,5))
    plt.hist(data, bins="fd")
    plt.title(f"Histogram of {var}")
    plt.xlabel(var)
    plt.ylabel("Frequency")
    plt.show()
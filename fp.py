# Name: Kamalakar Reddy
# UMBC ID: CJ88254
# Course: CMSC 462

import pandas as pd


df1 = pd.read_csv("clean_daily_weather.csv")
df2 = pd.read_csv("rain_hourly_clean.csv")


#Q2 Checking for duplicate date values
print("Duplicate date values:", df1["DATE"].duplicated().any())

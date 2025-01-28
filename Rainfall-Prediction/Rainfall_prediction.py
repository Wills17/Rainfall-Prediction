# import libraries

import numpy as np
import pandas as pd
import seaborn as sns

# Load the dataset
"""Data Collection and Processing"""
data = pd.read_csv('./Rainfall.csv')
print(data)

# print the (unique) values in the "day" column
print(data["day"].unique())

# Drop the "day" column as it is not needed for the prediction
data = data.drop(columns=["day"])
print(data)

print(data.info())

# Remove leading or trailing whitespace from the column names
data.columns = data.columns.str.strip()

# Check for missing values in the dataset
print(data.isnull().sum())

print(data["winddirection"].unique())

data["winddirection"].fillna(data["winddirection"].mode()[0], inplace=True)
data["windspeed"].fillna(data["windspeed"].median(), inplace=True)

# Check for any remaining missing values in the dataset after filling
print(data.isnull().sum())

# Convert the "rainfall" column to numerical values: "no" to 0 and "yes" to 1
data["rainfall"] = data["rainfall"].map({"no": 0, "yes": 1})
print(data.head())


"""Exploratory Data Analysis"""
sns.set_style("whitegrid")

print(data.describe())
# import libraries

import numpy as np
import pandas as pd

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

# import libraries

import numpy as np



# 
"""Data Collection and Processing"""
data = pd.read_csv('Rainfall.csv')
print(data)

# print the unique values in the "day" column
print(data["day"].unique())

# Drop the "day" column as it is not needed for the prediction
data = data.drop(columns=["day"])
print(data)
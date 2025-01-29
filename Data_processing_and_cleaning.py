# import library
import pandas as pd

# Load the dataset
"""Data Collection and Processing"""
data = pd.read_csv('./Rainfall_dataset.csv')
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

# Print the unique values in the "winddirection" and "windspeed" columns
print(data["winddirection"].unique())
print(data["windspeed"].unique())

# fill missing values in the "winddirection" column with the mode value
data["winddirection"].fillna(data["winddirection"].mode()[0], inplace=True)

# fill missing values in the "windspeed" column with the median value
data["windspeed"].fillna(data["windspeed"].median(), inplace=True)

# Check for any remaining missing values in the dataset after filling
print(data.isnull().sum())

# convert the "rainfall" column to numerical values
data["rainfall"] = data["rainfall"].map({"no": 0, "yes": 1})
print(data.head())


# Save the preprocessed dataset to a new CSV file
data.to_csv("./preprocesssed_rainfall_dataset.csv", index=False)   
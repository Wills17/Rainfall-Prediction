# import the necessary libraries

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import resample

# Load the preprocessed dataset
# Ensure "Data_processing_and_cleaning.py" file has been run before runnning this 

data = pd.read_csv("./preprocesssed_rainfall_dataset.csv")


# Quick description of the dataset
print("\nQuick description:\n", data.describe())

# Print the columns in the dataset
print("\nColumns in dataset:\n", data.columns)


# plot correlation heatmap of the dataset
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Rainfall Dataset")
plt.show()

# plot boxplots for each column in the dataset
plt.figure(figsize=(18,10))
for i, column in enumerate(data.columns, 1):
    
    # Create a subplot for each column
    plt.subplot(4, 3, i)
    sns.boxplot(data[column])
    plt.title(f"Boxplot of {column}")
plt.tight_layout()
plt.show()

# Drop unnecessary columns from the dataset
data = data.drop(columns=["maxtemp", "mintemp", "dewpoint"], axis=1)


# Distribution of the "pressure" column
sns.histplot(data["pressure"], kde=True, color="purple")
plt.title("Distribution of Pressure")
plt.xlabel("Pressure")
plt.ylabel("Count") 
plt.show()

# Distribution of the "temperature" column
sns.histplot(data["temperature"], kde=True, color="skyblue")
plt.title("Distribution of Temperature")
plt.xlabel("Temperature")
plt.ylabel("Count")
plt.show()

# Distribution of the "humidity" column
sns.histplot(data["humidity"], kde=True, color="red")
plt.title("Distribution of Humidity")
plt.xlabel("Humidity")
plt.ylabel("Count")
plt.show()

# Distribution of the "windspeed" column
sns.histplot(data["windspeed"], kde=True, color="green")
plt.title("Distribution of Wind Speed")
plt.xlabel("Windspeed")
plt.ylabel("Count")
plt.show()


# Print the value counts of the "rainfall" column
print(data["rainfall"].value_counts()) 


# separate "rainfall" column into minority and majority classes
data_minority = data[data["rainfall"] == 0]
data_majority = data[data["rainfall"] == 1]

# shape of the majority and minority classes
print("Shape of majority class (rainfall=1):", data_majority.shape)
print("Shape of minority class (rainfall=0):", data_minority.shape)

# Downsample the majority class
data_majority_downsampled = resample(data_majority, replace=False, n_samples=len(data_minority), random_state=42)

# shape of the downsampled majority class
print("Shape of downsampled majority class:", data_majority_downsampled.shape)

# combine minority class and downsampled majority class
data_downsampled = pd.concat([data_minority, data_majority_downsampled])
print(data_downsampled.shape)

# Print the downsampled dataset
print(data_downsampled)

# save the downsampled dataset to a new CSV file
data_downsampled.to_csv("./downsampled_rainfall_dataset.csv", index=False)
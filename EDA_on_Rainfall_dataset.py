# import the necessary libraries

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import resample
from Data_processing_and_cleaning import data


# Quick description of the dataset
print("\nQuick description:\n", data.describe())

# Print the columns in the dataset
print("\nColumns in dataset:\n", data.columns)


# plot correlation heatmap of the dataset
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Rainfall Dataset")
#plt.show()
plt.close()


# Set the style for seaborn plots
sns.set_style("whitegrid")


# plot boxplots for each column in the dataset
plt.figure(figsize=(18,10))
for i, column in enumerate(data.columns, 1):
    
    # Create a subplot for each column
    plt.subplot(4, 3, i)
    sns.boxplot(data[column])
    plt.title(f"Boxplot of {column}")
    
plt.tight_layout()
#plt.show()

# Drop unnecessary columns from the dataset
data = data.drop(columns=["maxtemp", "mintemp", "dewpoint"], axis=1)

"""# Plot the distribution of the "temperature" column
sns.histplot(data["temperature"], kde=True, color="skyblue")
plt.title("Distribution of Temperature")
plt.xlabel("Temperature")
plt.ylabel("Count")
#plt.show()

# Visualize the value counts of the "rainfall" column
plt.figure(figsize=(10,8))
sns.countplot(data["rainfall"], order=data["rainfall"].value_counts().index, palette="viridis")
plt.title("Value Counts of Rainfall")
plt.xlabel("Rainfall")
plt.ylabel("Count")
plt.xticks(rotation=90)
#plt.show()"""

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

print(data_downsampled["rainfall"].value_counts())


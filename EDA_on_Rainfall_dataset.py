# import the necessary libraries
from Data_processing_and_cleaning import data
import seaborn as sns
import matplotlib.pyplot as plt

print(data)

# Quick description of the dataset
print("Quick description", data.describe())

# Print the columns in the dataset
print("Columns in dataset:", data.columns)

plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Rainfall Dataset")
plt.show()
plt.close()


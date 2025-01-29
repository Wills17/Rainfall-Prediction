# import the necessary libraries
from Data_processing_and_cleaning import data
import seaborn as sns
import matplotlib.pyplot as plt


# Quick description of the dataset
print("Quick description:\n", data.describe())

# Print the columns in the dataset
print("Columns in dataset:\n", data.columns)


# plot correlation heatmap of the dataset
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Rainfall Dataset")
plt.show()
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
plt.show()



# Plot the distribution of the "temperature" column
sns.histplot(data["temparature"], kde=True, color="skyblue")
plt.title("Distribution of Temperature")
plt.xlabel("Temperature")
plt.ylabel("Count")
plt.show()


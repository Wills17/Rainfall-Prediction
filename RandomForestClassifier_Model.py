# import libraries

from sklearn.model_selection import train_test_split
from EDA_on_Rainfall_dataset import data_downsampled
from sklearn.ensemble import RandomForestClassifier


X = data_downsampled.drop(columns=["rainfall"], axis=1)
y = data_downsampled["rainfall"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
RFC_model = RandomForestClassifier(random_state=42)

"""Model Training"""

# Define the parameter grid for RandomForestClassifier
param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_features": ["auto", "sqrt", "log2"],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}


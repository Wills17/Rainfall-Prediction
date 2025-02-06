# import libraries

import numpy as np
import pandas as pd
import pickle  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the downsampled dataset
data_downsampled = pd.read_csv("./Datasets/downsampled_dataset.csv")

# split dataset into features and target variable
X = data_downsampled.drop(columns=["rainfall"], axis=1)
y = data_downsampled["rainfall"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
LR_model = LogisticRegression(random_state=42)

""" Training Model"""
# Define the parameter grid for LogisticRegression
param_grid_lr = {
    "C": [0.1, 1, 10, 100],
    "solver": ["liblinear", "saga"]
}

# GridSearch with cross-validation to find the best hyperparameters
GridSearchCV_LR = GridSearchCV(estimator=LR_model, param_grid=param_grid_lr, cv=5, n_jobs=-1, verbose=2)
GridSearchCV_LR.fit(X_train, y_train)

# output the best model from GridSearch
Best_LR_model = GridSearchCV_LR.best_estimator_
print("Best parameters for Logistic Regression: ", GridSearchCV_LR.best_params_)

"""Model Evaluation"""
# Perform cross-validation on the training set
cv_scores = cross_val_score(Best_LR_model, X_train, y_train, cv=5)
print("Cross-validation scores for Logistic Regression: ", cv_scores)
print("Mean cross-validation score for Logistic Regression: ", cv_scores.mean())

# Prediction the test set
y_pred = Best_LR_model.predict(X_test)

# evaluate model's accuracy on the test set
print("Test set Accuracy: ", accuracy_score(y_test, y_pred))
print("Test set Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
print("Test set Classification Report: \n", classification_report(y_test, y_pred))

# Save the model to a file
LR_model = {"model": Best_LR_model, "feature_names": X.columns.tolist()} 

with open("./Models/LogisticRegression_model.pkl", "wb") as file:
    pickle.dump(LR_model, file)

print("Model saved successfully!")

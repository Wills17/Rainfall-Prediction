# import libraries
import numpy as np
import pandas as pd
import pickle  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load the downsampled dataset
data_downsampled = pd.read_csv("./Datasets/downsampled_dataset.csv")


# split dataset into features and target variable
X = data_downsampled.drop(columns=["rainfall"], axis=1)
y = data_downsampled["rainfall"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
LR_model = LogisticRegression(random_state=42)


"""Training Model"""
# Define the parameter grid for LogisticRegression
param_grid_lr = {
    "penalty": ["l1", "l2", "elasticnet", "none"],
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    "max_iter": [100, 200, 300, 500]
}

# GridSearch with cross-validation to find the best hyperparameters
GridSearchCV_LR = GridSearchCV(estimator=LR_model, param_grid=param_grid_lr, cv=5, n_jobs=-1, verbose=2)
GridSearchCV_LR.fit(X_train, y_train)

# output the best model from GridSearch
Best_LR_model = GridSearchCV_LR.best_estimator_
print("\nBest parameters for Logistic Regression: ", GridSearchCV_LR.best_params_)


"""Model Evaluation"""
# Perform cross-validation on the training set
cv_scores = cross_val_score(Best_LR_model, X_train, y_train, cv=5)
print("\nCross-validation scores for Logistic Regression: ", cv_scores)
print("\nMean cross-validation score for Logistic Regression: ", cv_scores.mean())

# Prediction the test set
y_pred = Best_LR_model.predict(X_test)

# evaluate model's accuracy on the test set
print("\nTest set Accuracy: ", accuracy_score(y_test, y_pred))
print("\nTest set Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
print("\nTest set Classification Report: \n", classification_report(y_test, y_pred))


"""Test on Random data"""
# Prediction on random data
random_data = (1021.8, 10.4, 78, 88, 20, 50.0, 28.8)

# Reshape the data to match the model's expected input
random_data = np.array(random_data).reshape(1, -1)

# Make a prediction using the best Logistic Regression model output from GridSearch
prediction = Best_LR_model.predict(random_data)

if prediction == 0:
    print("Prediction result: ", "No Rainfall")
else:
    print("Prediction result: ", "Rainfall")
    

# Save the model to a file
LR_model = {"model": Best_LR_model, "feature_names": X.columns.tolist()} 

with open("./Prediction_LogisticRegression/LogisticRegression_model.pkl", "wb") as file:
    pickle.dump(LR_model, file)

print("Model saved successfully!")

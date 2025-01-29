# import libraries

import numpy as np
from sklearn.model_selection import train_test_split
from EDA_on_Rainfall_dataset import data_downsampled
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score



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

# GridSearch with cross-validation to find the best hyperparameters
GridSearchCV_RFC = GridSearchCV(estimator=RFC_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
GridSearchCV_RFC.fit(X_train, y_train)

# output the best model from GridSearch
best_RFC_model = GridSearchCV_RFC.best_estimator_
print("Best parameters for Random Forest: ", GridSearchCV_RFC.best_params_)


cv_scores = cross_val_score(best_RFC_model, X_train, y_train, cv = 5)
print("Cross-validation scores for Random Forest: ", cv_scores)
print("Mean cross-validation score for Random Forest: ", cv_scores.mean())
print("Mean cross-validation score for Random Forest: ", np.mean(cv_scores))


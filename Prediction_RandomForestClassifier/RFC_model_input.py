#import libraries

import pickle
import numpy as np


# Load the model from the file
with open("./Rainfall-Prediction/RandomForestClassifier_model.pkl", "rb") as file:
    RFC_model = pickle.load(file)
    
model = RFC_model["model"]
feature_names = RFC_model["feature_names"]

print("Model loaded successfully!\n")


print("Enter values for Prediction.")
def catcherror(user_input, feature):
    try:
        value = float(input(f"Enter value for {feature}: "))
        user_input.append(value)
    except ValueError:
        print(f"\nInvalid input for {feature}. Please enter a numeric value.")
        catcherror(user_input, feature)

while True:
    user_input = []
    for feature in feature_names:
        feature = feature.capitalize()
        
        # catch error in input
        catcherror(user_input, feature)

    user_input = np.array(user_input).reshape(1, -1)

    prediction = model.predict(user_input)    
    if prediction == 0:
        print("Prediction result: ", "No Rainfall\n")
    else:
        print("Prediction result: ", "Rainfall\n")
    
    repeat = input("Do you want to enter new values? (yes/no): \n").strip().lower()
    if repeat != 'yes':
        break
    
print("Exiting the model. Goodbye!")

#import libraries
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load the model from the file
with open("./Prediction_XGBClassifier/XGBClassifier_model.pkl", "rb") as file:
    XGB_model = pickle.load(file)
    
model = XGB_model["model"]
feature_names = XGB_model["feature_names"]

print("Model loaded successfully!\n")

print("Enter values for Prediction.")
def catcherror(user_input, feature):
    try:
        value = float(input(f"Enter value for {feature}: "))
        user_input.append(value)
        print("Valid input!\n")
    except ValueError:
        print(f"Invalid input for {feature}. Please enter a numeric value.")
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
    if repeat != 'yes' and repeat != 'y':
        break
    
print("Exiting the model. Goodbye!")
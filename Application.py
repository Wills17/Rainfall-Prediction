#import libraries
from flask import Flask, request, jsonify
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import random


app = Flask(__name__)

@app.route("/")
def hello():
    return "Welcome to the Rainfall Prediction Application"


"""@app.route("/predict", methods=["GET"])
def predict():
    def get_model():
        random_model = random.choice([0, 1])
        if random_model == 0:
        # Load the model from the file
            with open("./Prediction_LogisticRegression/LogisticRegression_model.pkl", "rb") as file:
                print("You are using Logistic Regression model.")
                use_model = pickle.load(file)
        else:
            with open("./Prediction_RandomForest/RandomForest_model.pkl", "rb") as file:
                print("You are using Random Forest model.")
                use_model = pickle.load(file)
        return use_model
            
    use_model = get_model()
    model = use_model["model"]
    feature_names = use_model["feature_names"]

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
        if repeat != 'yes' or repeat != 'y':
            break
        
    print("Exiting the model. Goodbye!")
"""
with open("./Prediction_LogisticRegression/LogisticRegression_model.pkl", "rb") as file:
    print("You are using Logistic Regression model.")
    use_model = pickle.load(file)
                
@app.route("/predict", methods=["GET"])
def predict():
    temp = float(request.args.get("temp"))
    rainfall = float(request.args.get("rainfall"))
    humidity = float(request.args.get("humidity"))
    wind_speed = float(request.args.get("wind_speed"))
    wind_dir = float(request.args.get("wind_dir"))
    pressure = float(request.args.get("pressure"))
    visibility = float(request.args.get("visibility"))

    # Perform prediction
    prediction = use_model.predict([[temp, rainfall, humidity, wind_speed, wind_dir, pressure, visibility]])

    # Return the result as JSON
    return jsonify({   
        "prediction": "Rainfall" if prediction[0] == 1 else "No Rainfall"
    })
    
if __name__ == "__main__":
    app.run(debug=True)
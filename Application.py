# import libraries
import pickle
from flask import Flask, request, render_template
import numpy as np
import random

app = Flask(__name__)

# Load either of the pre-trained model randomly
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
                
# Extract the actual model object from the loaded dictionary
use_model = use_model.get('model')


# Home route to render an input form
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

# Prediction route to handle form submissions and make predictions
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            # Extract data from the form
            rainfall_features = [float(x) for x in request.form.values()]
            features_array = np.array(rainfall_features).reshape(1, -1)
            
            # Make prediction using the model
            prediction = use_model.predict(features_array)
            
            # Return the result
            result = "Rainfall Expected" if prediction[0] == 1 else "No Rainfall Expected"
            return render_template('predict.html', prediction_text=result)
            
        except ValueError:
            return render_template('predict.html', error="Invalid input! Please enter numbers only.")
    else:
        return render_template('predict.html')
            
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

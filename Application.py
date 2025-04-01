#import libraries
from flask import Flask, request, jsonify
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import random



import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the pre-trained model (you can use your own model here)
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
                
# Home route to render an input form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route to handle form submissions and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract data from the form
        rainfall_features = [float(x) for x in request.form.values()]
        features_array = np.array(rainfall_features).reshape(1, -1)
        
        # Make prediction using the model
        prediction = use_model.predict(features_array)
        
        # Return the result
        return render_template('index.html', prediction_text=f'Predicted Rainfall: {prediction[0]} mm')

if __name__ == "__main__":
    app.run(debug=True)
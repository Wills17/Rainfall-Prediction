#import libraries

import pickle
import numpy as np

# Load the model from the file
with open("RandomForestClassifier_model.pkl", "rb") as file:
    RFC_model = pickle.load(file)
    
model = RFC_model["model"]
feature_names = RFC_model["feature_names"]

print("Model loaded successfully!")

print(feature_names)

new_data = (1003.8, 15.4, 18, 78, 60, 30.0, 24.8)
new_data = np.array(new_data).reshape(1, -1)

prediction = model.predict(new_data)    
if prediction == 0:
    print("Prediction result: ", "No Rainfall")
else:
    print("Prediction result: ", "Rainfall")
    


from flask import Flask
import pickle as pkl


app = Flask(__name__)
model_1 = pkl.load(open("./Prediction_LogisticRegression/LogisticRegression_model.pkl", "rb"))
model_2 = pkl.load(open("./Prediction_RandomForest/RandomForest_model.pkl", "rb"))


@app.route("/")
def hello():
    return "Welcome to the Rainfall Prediction Application"

@app.route("/predict", methods=["GET"])
def predict(model, temp, rainfall, humidity, wind_speed, wind_dir, pressure, visibility):
    if model == "LogisticRegression":
        prediction = model_1["model"].predict([[temp, rainfall, humidity, wind_speed, wind_dir, pressure, visibility]])
        if prediction == 0:
            return "Prediction result: No Rainfall"
        else:
            return "Prediction result: Rainfall"
    elif model == "RandomForest":
        prediction = model_2["model"].predict([[temp, rainfall, humidity, wind_speed, wind_dir, pressure, visibility]])
        if prediction == 0:
            return "Prediction result: No Rainfall"
        else:
            return "Prediction result: Rainfall"
    else:
        return "Invalid model name. Please choose either LogisticRegression or RandomForest"

if __name__ == "__main__":
    app.run(debug=True) 
# Rainfall Prediction with Various Machine Learning Models

This project aims to predict rainfall using various machine learning models. The dataset used contains weather-related features, and the target variable is whether it rained or not.

## Project Structure

- `Data_processing_and_cleaning.py`: Script for loading, cleaning, and preprocessing the dataset.
- `EDA_on_Rainfall_dataset.py`: Script for exploratory data analysis (EDA) on the dataset.
- `RandomForestClassifier_Model.py`: Script for training a Random Forest Classifier model.
- `RFC_model_input.py`: Script for loading the trained Random Forest model and making predictions based on user input.
- `LogisticRegression_Model.py`: Script for training a Logistic Regression model.
- `LR_model_input.py`: Script for loading the trained Logistic Regression model and making predictions based on user input.
- `Rainfall_dataset.csv`: The original dataset containing weather-related features and the target variable.

## Setup Instructions

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/Rainfall-prediction-with-various-Machine-Learning-models.git
    cd Rainfall-prediction-with-various-Machine-Learning-models
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Ensure you have the dataset file `Rainfall_dataset.csv` in the project directory.

## Usage

1. **Data Processing and Cleaning**:
    Run the script to clean and preprocess the dataset.
    ```sh
    python Data_processing_and_cleaning.py
    ```

2. **Exploratory Data Analysis (EDA)**:
    Perform EDA on the dataset to understand the data distribution and relationships.
    ```sh
    python EDA_on_Rainfall_dataset.py
    ```

3. **Model Training**:
    Train the Random Forest Classifier model using the preprocessed dataset.
    ```sh
    python RandomForestClassifier_Model.py
    ```
    Train the Logistic Regression model using the preprocessed dataset.
    ```sh
    python LogisticRegression_Model.py
    ```

4. **Model Prediction**:
    Use the trained Random Forest model to make predictions based on user input.
    ```sh
    python RFC_model_input.py
    ```
    Use the trained Logistic Regression model to make predictions based on user input.
    ```sh
    python LR_model_input.py
    ```

## Scripts Description

- **Data_processing_and_cleaning.py**:
    - Loads the dataset.
    - Cleans and preprocesses the data by handling missing values and dropping unnecessary columns.

- **EDA_on_Rainfall_dataset.py**:
    - Performs exploratory data analysis.
    - Visualizes data distributions and correlations.
    - Downsamples the majority class to balance the dataset.

- **RandomForestClassifier_Model.py**:
    - Trains a Random Forest Classifier model.
    - Uses GridSearchCV for hyperparameter tuning.
    - Evaluates the model and saves the best model to a file.

- **RFC_model_input.py**:
    - Loads the trained Random Forest model.
    - Takes user input for prediction.
    - Outputs the prediction result (Rainfall or No Rainfall).

- **LogisticRegression_Model.py**:
    - Trains a Logistic Regression model.
    - Uses GridSearchCV for hyperparameter tuning.
    - Evaluates the model and saves the best model to a file.

- **LR_model_input.py**:
    - Loads the trained Logistic Regression model.
    - Takes user input for prediction.
    - Outputs the prediction result (Rainfall or No Rainfall).

## Dataset

The dataset `Rainfall_dataset.csv` contains the following columns:
- `day`: Day of the observation.
- `pressure`: Atmospheric pressure.
- `maxtemp`: Maximum temperature.
- `temperature`: Average temperature.
- `mintemp`: Minimum temperature.
- `dewpoint`: Dew point temperature.
- `humidity`: Humidity percentage.
- `cloud`: Cloud cover percentage.
- `rainfall`: Target variable (yes/no).
- `sunshine`: Sunshine duration.
- `winddirection`: Wind direction.
- `windspeed`: Wind speed.

## License

This project is licensed under the MIT License.


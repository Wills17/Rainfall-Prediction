# Rainfall Prediction

This project aims to predict rainfall using multiple machine learning models. The dataset comprises weather-related features, with the target variable indicating whether it rained or not.

## Project Structure

The repository contains the following files and directories:

- **Datasets/**: Directory containing the dataset files used for training and evaluation.

- **Prediction_LogisticRegression/**: Directory containing scripts and models related to rainfall prediction using Logistic Regression.

- **Prediction_RandomForest/**: Directory containing scripts and models related to rainfall prediction using Random Forest.

- **templates/**: Directory containing HTML templates for the web application interface.

- **Application.py**: Main script to run the Flask web application for rainfall prediction.

- **Data_processing_and_cleaning.py**: Script for loading, cleaning, and preprocessing the dataset.

- **EDA_on_Rainfall_dataset.py**: Script for performing exploratory data analysis (EDA) on the dataset.


## Setup and Installation

To set up the project on your local machine, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Wills17/Rainfall-prediction.git
   ```


2. **Navigate to the Project Directory**:

   ```bash
   cd Rainfall-prediction
   ```


3. **Install Required Dependencies**:

   Ensure you have Python installed. It's recommended to use a virtual environment:

   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows, use 'env\Scripts\activate'
   ```

   Then, install the necessary packages:

   ```bash
   pip install -r requirements.txt
   ```


## Data Processing and Cleaning

The `Data_processing_and_cleaning.py` script handles data loading and preprocessing:

- **Loading Data**: Reads the dataset from the `Datasets/` directory.

- **Handling Missing Values**: Identifies and addresses missing data through imputation or removal.

- **Encoding Categorical Variables**: Converts categorical features into numerical representations suitable for machine learning models.

- **Feature Scaling**: Standardizes or normalizes numerical features to ensure consistent scaling.

## Exploratory Data Analysis (EDA)

The `EDA_on_Rainfall_dataset.py` script provides insights into the dataset:

- **Statistical Summaries**: Generates descriptive statistics for numerical and categorical features.

- **Visualizations**: Creates plots such as histograms, scatter plots, and box plots to visualize data distributions and relationships.

- **Correlation Analysis**: Examines correlations between features to identify potential predictors.

## Model Training and Prediction

Two primary models are implemented:

1. **Random Forest Classifier**:

   - *Training*: The `Prediction_RandomForest/RandomForest_Model.py` script trains a Random Forest model using the preprocessed data.

   - *Prediction*: The `Prediction_RandomForest/RFC_model_input.py` script loads the trained model and makes predictions based on user input.

2. **Logistic Regression**:

   - *Training*: The `Prediction_LogisticRegression/LogisticRegression_Model.py` script trains a Logistic Regression model.

   - *Prediction*: The `Prediction_LogisticRegression/LR_model_input.py` script loads the trained model and makes predictions based on user input.

## Web Application

A Flask-based web application is provided for user interaction:

- **Running the Application**:

   ```bash
   python Application.py
   ```

   This starts a local web server, typically accessible at `http://127.0.0.1:5000/`.

- **Using the Application**:

   Users can input weather-related features through the web interface. The application processes this input and returns a prediction indicating the likelihood of rainfall.

## How to Use

1. **Data Preparation**:

   - Place your dataset files in the `Datasets/` directory. Ensure they match the expected format used in the preprocessing scripts.

2. **Preprocess the Data**:

   - Run the `Data_processing_and_cleaning.py` script to clean and prepare the data for modeling.

3. **Perform EDA (Optional but Recommended)**:

   - Execute the `EDA_on_Rainfall_dataset.py` script to gain insights into the data and inform feature selection.

4. **Train the Models**:

   - For Random Forest: Run `Prediction_RandomForest/RandomForest_Model.py`.

   - For Logistic Regression: Run `Prediction_LogisticRegression/LogisticRegression_Model.py`.

5. **Make Predictions**:

   - Use the respective model input scripts (`RFC_model_input.py` or `LR_model_input.py`) to input new data and obtain predictions.

6. **Launch the Web Application**:

   - Run `Application.py` to start the Flask web interface for more user-friendly interactions.

## Demo Website

Explore the live demo of the rainfall prediction web application:

[Rainfall Prediction Demo](https://wills17.pythonanywhere.com/)

- Use the demo to input weather-related features and see real-time predictions.
- Note: The demo is hosted on a free-tier server, so it may experience occasional downtime or slower response times.



## Notes

- Ensure all dependencies listed in `requirements.txt` are installed.

- Modify the scripts as needed to accommodate different datasets or additional features.

For further assistance or questions, please refer to the repository's issues section or contact the project maintainer. Thank you.

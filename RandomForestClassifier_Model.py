# import libraries

from sklearn.model_selection import train_test_split
from EDA_on_Rainfall_dataset import data_downsampled


X = data_downsampled.drop(columns=["rainfall"], axis=1)
y = data_downsampled["rainfall"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

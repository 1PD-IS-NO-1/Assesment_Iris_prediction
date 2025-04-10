# model.py

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib  # Used to save the model

# model.py

import joblib

def load_model():
    model = joblib.load('model.pkl')
    label_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    return model, label_map

def predict_class(model, label_map, sepal_length, sepal_width):
    prediction = model.predict([[sepal_length, sepal_width]])
    return label_map[int(prediction[0])]

# Load Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Using sepal length and width only
y = iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.pkl')

print("✅ Model trained and saved as model.pkl")


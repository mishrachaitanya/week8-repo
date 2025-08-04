import pandas as pd
import numpy as np
import joblib
import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

@pytest.fixture(scope="module")
def test_data_and_model():
    df = pd.read_csv("iris.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = joblib.load("model.joblib")
    return model, X_test, y_test

def test_accuracy_threshold(test_data_and_model):
    model, X_test, y_test = test_data_and_model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test 1 - Accuracy: {acc:.4f}")
    assert acc >= 0.7, f"Accuracy too low: {acc:.4f}"

def test_class_coverage(test_data_and_model):
    model, X_test, _ = test_data_and_model
    y_pred = model.predict(X_test)
    unique_classes = np.unique(y_pred)
    print(f"Test 2 - Unique classes predicted: {unique_classes}")
    assert set(unique_classes) == {0, 1, 2}, f"Not all classes predicted: {unique_classes}"


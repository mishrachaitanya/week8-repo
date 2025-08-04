import pandas as pd
import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

# Load training data
train_df = pd.read_csv("train.csv")
X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]

# Load validation set
X_val = pd.read_csv("X_val.csv")
y_val = pd.read_csv("y_val.csv").squeeze()  # ensure it's a Series

# Encode target labels if not numeric
if y_train.dtype == 'object' or y_val.dtype == 'object':
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)

# Simulate 5 epochs of training
metrics = []
for epoch in range(1, 6):
    model = LogisticRegression(max_iter=epoch * 50)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    acc = accuracy_score(y_val, y_pred)
    loss = log_loss(y_val, y_proba)
    metrics.append([epoch, acc, loss])

# Write metrics for CML
with open("metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "accuracy", "loss"])
    writer.writerows(metrics)

# Also print final results to stdout for train_output.txt
print("Final Epoch Accuracy:", metrics[-1][1])

print("Final Epoch Loss:", metrics[-1][2])


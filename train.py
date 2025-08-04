import pandas as pd
import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('iris.csv')
X = df.drop(columns=['target'])
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simulate 5 epochs
metrics = []
for epoch in range(1, 6):
    model = LogisticRegression(max_iter=epoch * 50)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)
    metrics.append([epoch, acc, loss])

# Write metrics.csv for CML
with open('metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'accuracy', 'loss'])
    writer.writerows(metrics)


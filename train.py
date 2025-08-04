# train.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv('iris.csv')
X = df.drop(columns=['target'])
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training (simulated epochs)
report_lines = ["# Iris Model Training Report\n\n", "| Epoch | Accuracy | Log Loss |\n", "|-------|----------|----------|\n"]
epochs = 5
for epoch in range(1, epochs + 1):
    model = LogisticRegression(max_iter=epoch * 50)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)
    report_lines.append(f"| {epoch} | {acc:.4f} | {loss:.4f} |\n")

# Save model and report
joblib.dump(model, 'model.joblib')
with open('report.md', 'w') as f:
    f.writelines(report_lines)


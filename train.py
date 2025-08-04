import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
import joblib
import csv

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('iris.csv')

if 'target' not in df.columns:
    raise ValueError("Expected column 'target' not found in iris.csv")

X = df.drop(columns=['target'])
y = df['target']

# Split data into training and testing sets
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model over simulated epochs and collect metrics
metrics = []
print("Starting training...")
for epoch in range(1, 6):  # Simulate 5 epochs with increasing max_iter
    model = LogisticRegression(max_iter=epoch * 50, solver='lbfgs', multi_class='auto')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)
    
    metrics.append([epoch, acc, loss])
    print(f"Epoch {epoch}: Accuracy={acc:.4f}, Log Loss={loss:.4f}")

# Save metrics to CSV
print("Saving metrics to metrics.csv...")
with open('metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'accuracy', 'loss'])
    writer.writerows(metrics)

# Save the trained model
print("Saving model to model.joblib...")
joblib.dump(model, 'model.joblib')

print("Training completed successfully.")


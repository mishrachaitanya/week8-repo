import pandas as pd
import numpy as np
import argparse
import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from joblib import dump

# Parse the poisoning percentage
parser = argparse.ArgumentParser(description='Train Iris Model with Label Poisoning')
parser.add_argument('poison_percent', type=int, help='Percentage of labels to poison (0, 5, 10, 50)')
args = parser.parse_args()
poison_percent = args.poison_percent

# Load the original data
df = pd.read_csv("iris.csv")
X = df.drop(columns=["target"])
y = df["target"].copy()

# Poison the labels
if poison_percent > 0:
    np.random.seed(42)
    n_poison = int(len(y) * poison_percent / 100)
    poison_indices = np.random.choice(y.index, size=n_poison, replace=False)
    for idx in poison_indices:
        original_label = y[idx]
        possible_labels = list(set(y.unique()) - {original_label})
        y.at[idx] = np.random.choice(possible_labels)

# Split into train/validation (80/20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and collect metrics for 5 epochs
metrics = []
for epoch in range(1, 6):
    model = LogisticRegression(max_iter=epoch * 50)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    acc = accuracy_score(y_val, y_pred)
    loss = log_loss(y_val, y_proba)
    metrics.append([epoch, acc, loss])

# Save model
dump(model, "model.joblib")

# Save metrics
with open("metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "accuracy", "loss"])
    writer.writerows(metrics)

# Plot metrics
epochs = [m[0] for m in metrics]
accs = [m[1] for m in metrics]
losses = [m[2] for m in metrics]

plt.figure(figsize=(8, 5))
plt.plot(epochs, accs, marker='o', label="Accuracy")
plt.plot(epochs, losses, marker='x', label="Loss")
plt.title(f"Metrics over Epochs ({poison_percent}% Poisoned)")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True)
plt.savefig("metrics_plot.png")


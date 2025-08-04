import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
df = pd.read_csv("metrics.csv")

# Plot
plt.figure(figsize=(10, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(df["epoch"], df["accuracy"], marker='o', color='green')
plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(df["epoch"], df["loss"], marker='o', color='red')
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.tight_layout()
plt.savefig("metrics_plot.png")
plt.show()


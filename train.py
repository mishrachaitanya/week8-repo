import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
df = pd.read_csv("metrics.csv")

# Plot accuracy and loss
fig, ax1 = plt.subplots()

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy', color='tab:blue')
ax1.plot(df['epoch'], df['accuracy'], marker='o', color='tab:blue', label='Accuracy')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Loss', color='tab:red')  
ax2.plot(df['epoch'], df['loss'], marker='x', color='tab:red', label='Loss')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
plt.title('Accuracy and Loss over Epochs')
plt.savefig("plot.png")

import pandas as pd
from sklearn.model_selection import train_test_split
import random

# Load full iris dataset
df = pd.read_csv("iris.csv")

# Split into train and validation
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Save validation set
X_val.to_csv("X_val.csv", index=False)
y_val.to_csv("y_val.csv", index=False)

# Combine X_train and y_train
train_df = X_train.copy()
train_df["target"] = y_train
train_df.to_csv("train.csv", index=False)

# Get unique labels for flipping
labels = train_df["target"].unique()

# Poisoning function
def create_poisoned_copy(df, poison_percent, output_filename):
    df_copy = df.copy()
    num_poison = int(poison_percent * len(df_copy))
    poison_indices = random.sample(list(df_copy.index), num_poison)
    for idx in poison_indices:
        current_label = df_copy.at[idx, "target"]
        other_labels = [label for label in labels if label != current_label]
        df_copy.at[idx, "target"] = random.choice(other_labels)
    df_copy.to_csv(output_filename, index=False)
    print(f"✅ Created: {output_filename}")

# Create poisoned datasets at 5%, 10%, and 50%
create_poisoned_copy(train_df, 0.05, "poisoned_train_5.csv")
create_poisoned_copy(train_df, 0.10, "poisoned_train_10.csv")
create_poisoned_copy(train_df, 0.50, "poisoned_train_50.csv")


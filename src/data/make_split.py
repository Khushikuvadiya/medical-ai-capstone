from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Dataset is now inside the project
dataset_path = Path("data/raw/chest_xray")

# CSV output folder
output_path = Path("data")

records = []

for class_name in ["NORMAL", "PNEUMONIA"]:
    class_folder = dataset_path / "train" / class_name

    for image_path in class_folder.iterdir():
        if image_path.is_file():
            records.append({
                "image_path": str(image_path),
                "label": class_name
            })

# Create DataFrame
df = pd.DataFrame(records)

# Create stratified 80/20 train-validation split
train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# Make sure data folder exists
output_path.mkdir(exist_ok=True)

# Save split CSV files
train_df.to_csv(
    output_path / "train_split.csv",
    index=False
)

val_df.to_csv(
    output_path / "val_split.csv",
    index=False
)

# Print summary
print("\nDATASET SPLIT SUMMARY")
print("-" * 40)

print("Total original training images:", len(df))
print("New training images:", len(train_df))
print("New validation images:", len(val_df))

print("\nTraining class counts:")
print(train_df["label"].value_counts())

print("\nValidation class counts:")
print(val_df["label"].value_counts())

print("-" * 40)
print("Split files created successfully.")
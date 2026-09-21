from pathlib import Path

import pandas as pd


dataset_path = Path("data/raw/chest_xray")
test_path = dataset_path / "test"

records = []

for class_name in ["NORMAL", "PNEUMONIA"]:

    class_folder = test_path / class_name

    for image_path in class_folder.iterdir():

        if image_path.is_file():

            records.append({
                "image_path": str(image_path),
                "label": class_name
            })


test_df = pd.DataFrame(records)

test_df.to_csv(
    "data/test_split.csv",
    index=False
)

print("\nTEST DATASET")
print("-" * 40)

print("Total test images:", len(test_df))

print("\nClass counts:")
print(test_df["label"].value_counts())

print("-" * 40)

print("Saved: data/test_split.csv")
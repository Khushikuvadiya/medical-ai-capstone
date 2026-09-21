from pathlib import Path

# Dataset location inside the project
dataset_path = Path("data/raw/chest_xray")

folders = [
    dataset_path / "train" / "NORMAL",
    dataset_path / "train" / "PNEUMONIA",
    dataset_path / "val" / "NORMAL",
    dataset_path / "val" / "PNEUMONIA",
    dataset_path / "test" / "NORMAL",
    dataset_path / "test" / "PNEUMONIA",
]

print("\nCHEST X-RAY DATASET CHECK")
print("-" * 40)

total_images = 0

for folder in folders:

    if not folder.exists():
        print(f"ERROR: Folder not found -> {folder}")
        continue

    images = [file for file in folder.iterdir() if file.is_file()]

    count = len(images)
    total_images += count

    print(f"{folder}: {count} images")

print("-" * 40)
print(f"Total images: {total_images}")
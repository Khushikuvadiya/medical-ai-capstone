from dataset import ChestXrayDataset

dataset = ChestXrayDataset("data/train_split.csv")

print("Total images:", len(dataset))

image, label = dataset[0]

print("Image shape:", image.shape)
print("Label:", label)
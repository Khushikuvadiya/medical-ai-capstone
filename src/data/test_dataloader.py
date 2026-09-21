from torch.utils.data import DataLoader
from dataset import ChestXrayDataset

dataset = ChestXrayDataset("data/train_split.csv")

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

images, labels = next(iter(loader))

print("Number of images in dataset:", len(dataset))
print("Batch image shape:", images.shape)
print("Batch labels shape:", labels.shape)
print("Labels:", labels)
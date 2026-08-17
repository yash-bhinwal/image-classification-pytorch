from torchvision import datasets


train_dataset = datasets.CIFAR10(
    root="data/raw",
    train=True,
    download=False
)

test_dataset = datasets.CIFAR10(
    root="data/raw",
    train=False,
    download=False
)

print("Training samples:", len(train_dataset))
print("Test samples:", len(test_dataset))
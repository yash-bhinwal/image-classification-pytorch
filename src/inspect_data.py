from torchvision import datasets, transforms
import matplotlib.pyplot as plt


transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root="data/raw",
    train=True,
    download=False,
    transform=transform
)

image, label = train_dataset[0]

print("Image type:", type(image))
print("Label type:", type(label))
print("Image shape:", image.shape)
print("Image dtype:", image.dtype)
print("Minimum pixel value:", image.min())
print("Maximum pixel value:", image.max())
print("Label:", label)

plt.imshow(image.permute(1, 2, 0))
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()
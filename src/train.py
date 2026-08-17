import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleMLP


transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root="data/raw",
    train=True,
    download=False,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="data/raw",
    train=False,
    download=False,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

model = SimpleMLP()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

num_epochs = 5

for epoch in range(num_epochs):

    total_loss = 0

    for images, labels in train_loader:

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = loss_fn(outputs, labels)

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Add this batch's loss
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Average Loss: {average_loss:.4f}"
    )

# Training accuracy

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in train_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

train_accuracy = 100 * correct / total

print(f"Training Accuracy: {train_accuracy:.2f}%")


# Test accuracy

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total

print(f"Test Accuracy: {test_accuracy:.2f}%")

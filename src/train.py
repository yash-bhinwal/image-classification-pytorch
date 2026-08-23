import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import CNN


# -------------------------
# Transforms
# -------------------------

# No data augmentation
train_transform = transforms.ToTensor()
test_transform = transforms.ToTensor()


# -------------------------
# Datasets
# -------------------------

train_full_dataset = datasets.CIFAR10(
    root="data/raw",
    train=True,
    download=False,
    transform=train_transform
)

val_full_dataset = datasets.CIFAR10(
    root="data/raw",
    train=True,
    download=False,
    transform=test_transform
)

test_dataset = datasets.CIFAR10(
    root="data/raw",
    train=False,
    download=False,
    transform=test_transform
)


# -------------------------
# Train / Validation Split
# -------------------------

generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    len(train_full_dataset),
    generator=generator
)

train_indices = indices[:45000]
val_indices = indices[45000:]


train_dataset = Subset(
    train_full_dataset,
    train_indices
)

val_dataset = Subset(
    val_full_dataset,
    val_indices
)


# -------------------------
# DataLoaders
# -------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# -------------------------
# Model
# -------------------------

model = CNN()


# -------------------------
# Loss
# -------------------------

loss_fn = nn.CrossEntropyLoss()


# -------------------------
# Optimizer
# -------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)


# -------------------------
# Learning Rate Scheduler
# -------------------------

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=2,
    gamma=0.5
)


# -------------------------
# Training
# -------------------------

num_epochs = 5

for epoch in range(num_epochs):

    model.train()

    total_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = loss_fn(outputs, labels)

        # Predictions
        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)

        train_correct += (
            predicted == labels
        ).sum().item()

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Track loss
        total_loss += loss.item()


    # -------------------------
    # Training Metrics
    # -------------------------

    average_loss = (
        total_loss / len(train_loader)
    )

    train_accuracy = (
        100 * train_correct / train_total
    )


    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_accuracy = (
        100 * val_correct / val_total
    )


    # -------------------------
    # Current Learning Rate
    # -------------------------

    current_lr = optimizer.param_groups[0]["lr"]


    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Average Loss: {average_loss:.4f}, "
        f"Training Accuracy: {train_accuracy:.2f}%, "
        f"Validation Accuracy: {val_accuracy:.2f}%, "
        f"Learning Rate: {current_lr:.6f}"
    )


    # -------------------------
    # Update Learning Rate
    # -------------------------

    scheduler.step()


# -------------------------
# Final Test Evaluation
# -------------------------

model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        test_total += labels.size(0)

        test_correct += (
            predicted == labels
        ).sum().item()


test_accuracy = (
    100 * test_correct / test_total
)


print(
    f"Final Test Accuracy: "
    f"{test_accuracy:.2f}%"
)
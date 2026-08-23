import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import CNN


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

MAX_EPOCHS = 30

# Early stopping:
# stop after 5 consecutive epochs without validation improvement
EARLY_STOPPING_PATIENCE = 5


# ============================================================
# Transforms
# ============================================================

# No augmentation for this experiment
train_transform = transforms.ToTensor()
test_transform = transforms.ToTensor()


# ============================================================
# Datasets
# ============================================================

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


# ============================================================
# Train / Validation Split
# ============================================================

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


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Model
# ============================================================

model = CNN()


# ============================================================
# Loss
# ============================================================

loss_fn = nn.CrossEntropyLoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# Learning Rate Scheduler
# ============================================================

# If validation loss stops improving,
# reduce the learning rate.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# Checkpoint Setup
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

checkpoint_path = "models/best_model.pth"

best_val_accuracy = 0.0

epochs_without_improvement = 0


# ============================================================
# Training
# ============================================================

for epoch in range(MAX_EPOCHS):

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    total_train_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        # Clear gradients from previous batch
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = loss_fn(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # Track training loss
        total_train_loss += loss.item()

        # Calculate predictions
        _, predicted = torch.max(
            outputs,
            1
        )

        train_total += labels.size(0)

        train_correct += (
            predicted == labels
        ).sum().item()


    average_train_loss = (
        total_train_loss /
        len(train_loader)
    )

    train_accuracy = (
        100 *
        train_correct /
        train_total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    total_val_loss = 0

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            outputs = model(images)

            loss = loss_fn(
                outputs,
                labels
            )

            total_val_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    average_val_loss = (
        total_val_loss /
        len(val_loader)
    )

    val_accuracy = (
        100 *
        val_correct /
        val_total
    )


    # --------------------------------------------------------
    # Display Metrics
    # --------------------------------------------------------

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    print(
        f"\nEpoch [{epoch + 1}/{MAX_EPOCHS}]"
    )

    print(
        f"Train Loss: {average_train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss: {average_val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

    print(
        f"Learning Rate: {current_lr:.6f}"
    )


    # --------------------------------------------------------
    # Learning Rate Scheduler
    # --------------------------------------------------------

    # ReduceLROnPlateau watches validation LOSS.
    scheduler.step(
        average_val_loss
    )


    # --------------------------------------------------------
    # Checkpointing
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        print(
            f"✓ New best model saved "
            f"(Val Accuracy: "
            f"{best_val_accuracy:.2f}%)"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No validation accuracy improvement "
            f"for {epochs_without_improvement} "
            f"epoch(s)."
        )


    # --------------------------------------------------------
    # Early Stopping
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# Load Best Model
# ============================================================

print(
    "\nLoading best checkpoint..."
)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        weights_only=True
    )
)

model.eval()


# ============================================================
# Final Test Evaluation
# ============================================================

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
    100 *
    test_correct /
    test_total
)


print("\n==============================")
print("FINAL RESULTS")
print("==============================")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Final Test Accuracy: "
    f"{test_accuracy:.2f}%"
)
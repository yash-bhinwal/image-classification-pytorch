import os
import random

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import CNN


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2


# ============================================================
# MLflow Experiment
# ============================================================

mlflow.set_experiment("cifar10-cnn")


# ============================================================
# Transforms
# ============================================================

# Training augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(
        32,
        padding=4
    ),
    transforms.ToTensor()
])

# Validation / test should NOT use random augmentation
test_transform = transforms.ToTensor()


# ============================================================
# Datasets
# ============================================================

# Training version of CIFAR-10
# Uses augmentation
train_full_dataset = datasets.CIFAR10(
    root="data/raw",
    train=True,
    download=False,
    transform=train_transform
)

# Separate view of the same 50,000 CIFAR-10 training images
# Used for validation WITHOUT augmentation
val_full_dataset = datasets.CIFAR10(
    root="data/raw",
    train=True,
    download=False,
    transform=test_transform
)

# Final untouched CIFAR-10 test set
test_dataset = datasets.CIFAR10(
    root="data/raw",
    train=False,
    download=False,
    transform=test_transform
)


# ============================================================
# Train / Validation Split
# ============================================================

generator = torch.Generator().manual_seed(SEED)

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

train_generator = torch.Generator()
train_generator.manual_seed(SEED)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=train_generator
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

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=SCHEDULER_FACTOR,
    patience=SCHEDULER_PATIENCE
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
best_epoch = 0
epochs_without_improvement = 0


# ============================================================
# MLflow Run
# ============================================================

with mlflow.start_run(
    run_name="augmentation-crop-flip"
):

    # --------------------------------------------------------
    # Log Hyperparameters / Configuration
    # --------------------------------------------------------

    mlflow.log_params({
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "architecture": "CNN",
        "augmentation": "RandomCrop32Padding4+HorizontalFlip"
    })


    # ========================================================
    # Training Loop
    # ========================================================

    for epoch in range(MAX_EPOCHS):

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

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

            # Track loss
            total_train_loss += loss.item()

            # Predictions
            _, predicted = torch.max(
                outputs,
                1
            )

            train_total += labels.size(0)

            train_correct += (
                predicted == labels
            ).sum().item()


        # ----------------------------------------------------
        # Training Metrics
        # ----------------------------------------------------

        average_train_loss = (
            total_train_loss /
            len(train_loader)
        )

        train_accuracy = (
            100 *
            train_correct /
            train_total
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Validation Metrics
        # ----------------------------------------------------

        average_val_loss = (
            total_val_loss /
            len(val_loader)
        )

        val_accuracy = (
            100 *
            val_correct /
            val_total
        )


        # ----------------------------------------------------
        # Current Learning Rate
        # ----------------------------------------------------

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )


        # ----------------------------------------------------
        # Print Metrics
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # MLflow: Log Epoch Metrics
        # ----------------------------------------------------

        mlflow.log_metrics(
            {
                "train_loss": average_train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": average_val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": current_lr
            },
            step=epoch + 1
        )


        # ----------------------------------------------------
        # Learning Rate Scheduler
        # ----------------------------------------------------

        scheduler.step(
            average_val_loss
        )


        # ----------------------------------------------------
        # Checkpointing
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1

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


        # ----------------------------------------------------
        # Early Stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break


    # ========================================================
    # Load Best Checkpoint
    # ========================================================

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


    # ========================================================
    # Final Test Evaluation
    # ========================================================

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


    # ========================================================
    # MLflow: Final Metrics
    # ========================================================

    mlflow.log_metric(
        "best_val_accuracy",
        best_val_accuracy
    )

    mlflow.log_metric(
        "test_accuracy",
        test_accuracy
    )

    mlflow.log_metric(
        "best_epoch",
        best_epoch
    )


    # ========================================================
    # MLflow: Log Best Checkpoint
    # ========================================================

    mlflow.log_artifact(
        checkpoint_path,
        artifact_path="checkpoints"
    )


    # ========================================================
    # Final Results
    # ========================================================

    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")

    print(
        f"Best Epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy:.2f}%"
    )

    print(
        f"Final Test Accuracy: "
        f"{test_accuracy:.2f}%"
    )
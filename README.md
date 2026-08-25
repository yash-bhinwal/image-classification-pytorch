# Image Classification with PyTorch

A deep learning project exploring image classification using PyTorch.

## Project Status

🚧 In development

## Goal

Build and understand an end-to-end image classification pipeline using PyTorch, starting from a simple neural network and progressing toward convolutional neural networks and transfer learning.

## Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Jupyter

## Project Structure

```text
data/          Dataset files
notebooks/     Experiments and exploration
src/           Reusable project code
tests/         Tests
models/        Trained model artifacts

## Experiment Results

Experiments are tracked using MLflow.

| Experiment | Data Augmentation | Best Validation Accuracy | Test Accuracy |
|---|---|---:|---:|
| Baseline CNN | None | 81.42% | 81.09% |
| Augmented CNN | RandomCrop + HorizontalFlip | 85.38% | 85.38% |

Adding data augmentation improved test accuracy from **81.09% to 85.38%**.

This corresponds to an approximately **22.7% relative reduction in classification error**.

The experiments use:
- Adam optimizer
- Weight decay
- ReduceLROnPlateau learning-rate scheduling
- Model checkpointing
- Early stopping
- Fixed random seeds for reproducibility
- MLflow experiment tracking
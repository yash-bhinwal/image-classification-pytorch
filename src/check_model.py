import torch
import torch.nn as nn

from model import SimpleMLP


model = SimpleMLP()

x = torch.randn(1, 3, 32, 32)
y = torch.tensor([6])

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# -------------------------
# BEFORE UPDATE
# -------------------------

output = model(x)
loss = loss_fn(output, y)

print("Loss before update:", loss.item())


# -------------------------
# BACKPROPAGATION
# -------------------------

optimizer.zero_grad()

loss.backward()


# -------------------------
# UPDATE WEIGHTS
# -------------------------

optimizer.step()


# -------------------------
# AFTER UPDATE
# -------------------------

output = model(x)
loss = loss_fn(output, y)

print("Loss after update:", loss.item())

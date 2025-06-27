"""
Comparison of Efficient KAN, Fast KAN, and MLP models for solving a simple
first-order ODE using physics-informed neural networks (PINNs).
The ODE is dx/dt = -x + sin(t), with known analytic solution.
          
 @author: D. A. Kaltsas
 June 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython
import time
from models import MLP, effKAN, fastKAN


# Setup: Clear plots, GPU memory, set device and data type
plt.close('all')  # Close any open matplotlib windows

get_ipython().run_line_magic('matplotlib', 'qt')  

torch.cuda.empty_cache()  # Free any unused GPU memory
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
print(f"Using device: {device}")

# Define the ODE and its analytic solution
# Right-hand side of the ODE: dx/dt = -x + sin(t)
def ode_rhs(x, t):
    return -x + torch.sin(t)

# Known analytic solution for comparison
def x_analytic(t):
    return 0.5 * (np.sin(t) - np.cos(t)) + 1.5 * np.exp(-t)


# Generate training data
Ntrain=200
t_train_full = torch.linspace(0, 5, Ntrain).view(-1, 1)  # No .requires_grad_ here

x0 = torch.tensor([[1.0]])  # Initial condition: x(0) = 1
t0 = torch.tensor([[0.0]])  # Initial time


batch_size = 200  # You can change this
train_dataset = TensorDataset(t_train_full)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Instantiate models
kan_model_1 = effKAN([1, 10, 10, 1], grid_size=5, spline_order=3)
kan_model_2 = fastKAN([1, 10, 10, 1], use_layernorm=False)
mlp_model = MLP([1, 10, 10, 1])

# Set up optimizers
kan_optimizer_1 = torch.optim.Adam(kan_model_1.parameters(), lr=1e-3)
kan_optimizer_2 = torch.optim.Adam(kan_model_2.parameters(), lr=1e-3)
mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-3)

lambda_ic = 10.0  # Penalty for initial condition mismatch
epochs = 1000     # Number of training epochs


# Training loop
def train_model(model, optimizer, train_loader):
    for epoch in range(epochs):
        for (t_batch,) in train_loader:
            t_batch = t_batch.clone().detach().requires_grad_(True)

            optimizer.zero_grad()
            x_hat = model(t_batch)

            dx_dt = torch.autograd.grad(
                x_hat, t_batch,
                grad_outputs=torch.ones_like(x_hat),
                create_graph=True,
            )[0]

            res = dx_dt - ode_rhs(x_hat, t_batch)

            loss_phys = torch.mean(res**2)
            loss_ic = (model(t0) - x0)**2
            loss = loss_phys + lambda_ic * loss_ic

            loss.backward()
            optimizer.step()

        if epoch % 200 == 0:
            print(f"{model.__class__.__name__} Epoch {epoch}, Loss: {loss.item():.6f}")


# -----------------------------------------------------------------------------
# Train and time each model
# -----------------------------------------------------------------------------
print("Training Efficient KAN...")
start = time.time()
train_model(kan_model_1, kan_optimizer_1,train_loader)
time_kan_1 = time.time() - start

print("Training Fast KAN...")
start = time.time()
train_model(kan_model_2, kan_optimizer_2,train_loader)
time_kan_2 = time.time() - start

print("\nTraining MLP...")
start = time.time()
train_model(mlp_model, mlp_optimizer,train_loader)
time_mlp = time.time() - start


# Evaluate each model on fine time grid
t_eval = torch.linspace(0, 5, 500).view(-1, 1)
x_kan_pred_1 = kan_model_1(t_eval).detach().numpy()
x_kan_pred_2 = kan_model_2(t_eval).detach().numpy()
x_mlp_pred = mlp_model(t_eval).detach().numpy()
t_plot = t_eval.numpy().flatten()
x_true = x_analytic(t_plot)

# Plot predictions vs. analytic solution
plt.figure(figsize=(10, 6))
plt.plot(t_plot, x_true, 'b', linewidth=2, label="Analytic Solution")
plt.plot(t_plot, x_kan_pred_1, 'r', label="Efficient KAN")
plt.plot(t_plot, x_kan_pred_2, 'g--', label="Fast KAN")
plt.plot(t_plot, x_mlp_pred, 'm-.', label="MLP")
plt.title("Solution to dx/dt = -x + sin(t)")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.grid(True)
plt.show()

# Report training durations
print(f"\nExecution Time (training only):")
print(f"  Efficient KAN: {time_kan_1:.6f} seconds")
print(f"  Fast KAN:      {time_kan_2:.6f} seconds")
print(f"  MLP:           {time_mlp:.6f} seconds")

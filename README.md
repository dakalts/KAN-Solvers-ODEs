# Comparison of Efficient KAN, Fast KAN, and MLP Models for Solving a Simple ODE

This project implements and compares three neural network architectures:

1. Efficient KAN (Kolmogorov-Arnold Network) with  B-spline basis functions

2. Fast KAN with Gaussian Radial Basis Functions

3. Standard MLP (Multi-Layer Perceptron)

The comparison is done in the context of solving a simple first-order ODE using Physics-Informed Neural Networks (PINNs).

## We solve the differential equation:

$$\frac{dx}{dt} =-x(t) + sin(t)\,, $$

with initial condition:
$$x(0)=1\,.$$

The exact (analytic) solution is known and used as ground truth for evaluation:
$$x(t)=0.5(sin⁡(t)−cos⁡(t))+1.5e^t$$


## Requirements
- Python 3.8+
- Pytorch
- Matplotlib
- NumPy
- Math

This implemantation uses classes effKAN and fastKAN, taken from Efficient KAN repo: https://github.com/Blealtan/efficient-kan and the Fast-KAN repo: https://github.com/ZiyaoLi/fast-kan

## If you use this code, please cite:


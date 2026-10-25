
# Neural Networks with NumPy — Matrix Experiments

This repository contains my experiments in implementing neurons and neural networks from scratch using Python and NumPy.

These projects build upon my earlier scalar implementations and explore how neural-network computations can be expressed using arrays, matrices, and vectorized mathematical operations.

The goal is to understand how neural networks work internally, including forward propagation, loss calculation, gradient computation, and parameter updates, without relying on a machine-learning framework.

## Experiments

### 1. Single Neuron — `neuron_matrix.py`

A single neuron implemented using NumPy arrays and matrix operations.

- Uses two inputs and a weight matrix.
- Learns to approximate addition.
- Calculates prediction errors using squared error.
- Updates weights and bias during training.
- Tests the trained neuron on a new input.

This experiment explores how the scalar neuron can be represented using matrix operations.

### 2. Linear Neural Network — `neural_network_linear_matrix.py`

A neural network with three hidden neurons and one output neuron, implemented using NumPy.

- Learns to approximate addition.
- Implements forward propagation using matrix multiplication.
- Calculates gradients manually.
- Updates weights and biases using gradient descent.
- Uses no activation function in the hidden layer.

This experiment explores how multiple neurons can be connected and trained using matrix-based calculations.

### 3. Nonlinear Neural Network — `neural_network_non-linear_matrix.py`

A neural network with eight hidden neurons and one output neuron, using the Leaky ReLU activation function.

- Learns to approximate absolute subtraction: `abs(x1 - x2)`.
- Implements Leaky ReLU and its derivative.
- Calculates gradients using the chain rule.
- Updates weights and biases through gradient descent.
- Tests the trained network with a new input.

This experiment introduces activation functions into a matrix-based neural network.

### 4. Multilayer Neural Network — `multi_layer_neural_network.py`

A deeper neural network with three hidden layers containing 8, 16, and 8 neurons, respectively, followed by one output neuron.

- Uses NumPy matrix multiplication for forward propagation.
- Applies Leaky ReLU activation functions to the hidden layers.
- Implements backpropagation manually.
- Calculates gradients for weights and biases.
- Uses he weight initialization.
- Trains on a custom mathematical relationship involving absolute differences and sums.

This experiment explores how multiple hidden layers can be connected and trained using manually implemented backpropagation.

## Concepts Explored

- Artificial neurons
- NumPy arrays and matrix multiplication
- Weights and biases
- Forward propagation
- Training data and target values
- Squared error and loss calculation
- Learning rates and gradient descent
- Manual gradient computation
- Backpropagation
- The chain rule
- Activation functions and their derivatives
- Leaky ReLU
- Weight initialization
- Multilayer neural-network architecture

## Technologies

- Python 3
- NumPy

These experiments implement the neural-network calculations manually rather than using frameworks such as PyTorch or TensorFlow.

## Requirements

Python 3 and NumPy are required.

Install NumPy with:

```bash
python3 -m pip install numpy
```

## How to Run

Run any experiment independently from the repository directory:

```bash
python3 neuron_matrix.py
```

```bash
python3 neural_network_linear_matrix.py
```

```bash
python3 neural_network_non-linear_matrix.py
```

```bash
python3 multi_layer_neural_network.py
```

The experiments generate training data and initialize parameters within their scripts. Results may vary between runs when random initialization or random training data are used.

## Purpose

This repository documents my exploration of neural networks using NumPy and matrix mathematics.

By implementing the calculations myself, I want to understand how neurons, layers, activation functions, gradients, and backpropagation work internally.

These experiments form a step between my scalar implementations and the development of reusable neural-network frameworks.

This is an ongoing personal learning project, and the implementations may evolve as I gain a deeper understanding of neural networks.

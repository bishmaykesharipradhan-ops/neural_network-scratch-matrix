import numpy as np

def leaky_relu(x):
    return np.maximum(0.01 * x, x)

def leaky_relu_derivative(x):
    return np.where(x > 0, 1, 0.01)

# TRAINING PHASE
SCALE = 100

training_data = []
for _ in range(100):
    x = np.random.randint(1, 100, size=(1, 2))
    target = np.array([abs(x[0, 0] - x[0, 1])])
    training_data.append((x/SCALE, target/SCALE))   

learning_rate = 0.01

w = np.random.uniform(-1, 1, size=(2, 8))
bias = np.zeros((1, 8))

w_output = np.random.uniform(-1, 1, size=(8, 1))
output_bias = np.zeros((1, 1))

for epoch in range(10000):
    total_loss = 0.0

    for x, target in training_data:
        # Forward pass
        z = np.dot(x, w) + bias
        hidden_output = leaky_relu(z)

        old_output = np.dot(hidden_output, w_output) + output_bias

        error = target - old_output
        loss = error ** 2
        total_loss += loss

        # GRADIENTS
        d_loss = 2 * (old_output - target)
        
        gradient_w_output = d_loss * hidden_output.T
        gradient_bias_output = d_loss

        gradient_w = d_loss * w_output.T * leaky_relu_derivative(z) * x.T
        gradient_bias = d_loss * w_output.T * leaky_relu_derivative(z)

        # Update weights and biases
        w_output = w_output - learning_rate * gradient_w_output
        output_bias = output_bias - learning_rate * gradient_bias_output
        
        w = w - learning_rate * gradient_w
        bias = bias - learning_rate * gradient_bias

    average_loss = np.mean(total_loss) / len(training_data)


# NEW OUTPUT
for x, target in training_data:
    z = np.dot(x, w) + bias
    hidden_output = leaky_relu(z)
    output = np.dot(hidden_output, w_output) + output_bias

    loss = (target - output) ** 2
    average_loss = np.mean(loss)
    print("Input:", x * SCALE, "Target:", target * SCALE, "Prediction:", output * SCALE, "Loss:", average_loss)

# TESTING PHASE
x = np.array([[13, 7]]) / SCALE
target = np.array([[6]]) / SCALE

z = np.dot(x, w) + bias
hidden_output = leaky_relu(z)

output = np.dot(hidden_output, w_output) + output_bias

error = target - output
loss = error ** 2
average_loss = np.mean(loss)

print("Output:", output * SCALE, "Loss:", average_loss)

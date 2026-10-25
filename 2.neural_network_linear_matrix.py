import numpy as np

# TRAINING PHASE
SCALE = 100

training_data = []
for _ in range(100):
    x = np.random.randint(1, 100, size=(1, 2))
    target = np.array([x[0, 0] + x[0, 1]])
    training_data.append((x/SCALE, target/SCALE))   

learning_rate = 0.001

w = np.random.uniform(-1, 1, size=(2, 3))
bias = np.zeros((1, 3))

w_output = np.random.uniform(-1, 1, size=(3, 1))
output_bias = np.zeros((1, 1))
for epoch in range(10000):
    total_loss = 0.0

    for x, target in training_data:
        # Forward pass
        hidden_output = np.dot(x, w) + bias
        old_output = np.dot(hidden_output, w_output) + output_bias

        error = target - old_output
        loss = error ** 2
        total_loss += loss

        # GRADIENTS
        d_loss = 2 * (old_output - target)
        gradient_w_output = d_loss * hidden_output.T
        gradient_bias_output = d_loss

        gradient_w = d_loss * w_output.T * x.T
        gradient_bias = d_loss * w_output.T

        # Update weights and biases
        w_output = w_output - learning_rate * gradient_w_output
        output_bias = output_bias - learning_rate * gradient_bias_output
        
        w = w - learning_rate * gradient_w
        bias = bias - learning_rate * gradient_bias

    average_loss = np.mean(total_loss) / len(training_data)

# NEW OUTPUT
for x, target in training_data:
    hidden_output = np.dot(x, w) + bias
    output = np.dot(hidden_output, w_output) + output_bias

    loss = (target - output) ** 2
    average_loss = np.mean(loss)
    print("Input:", x * SCALE, "Target:", target * SCALE, "Prediction:", output * SCALE, "Loss:", average_loss)

# TESTING PHASE
x = np.array([[343, 7]]) / SCALE
target = np.array([[350]]) / SCALE

hidden_output = np.dot(x, w) + bias
output = np.dot(hidden_output, w_output) + output_bias
error = target - output
loss = error ** 2
average_loss = np.mean(loss)

print("Output:", output * SCALE, "Loss:", average_loss)

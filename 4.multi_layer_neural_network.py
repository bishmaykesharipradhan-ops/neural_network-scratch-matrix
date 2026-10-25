import numpy as np

def leaky_relu(x):
    return np.maximum(0.01 * x, x)

def leaky_relu_derivative(x):
    return np.where(x > 0, 1, 0.01)

np.random.seed(0)

# TRAINING PHASE
SCALE = 100

training_data = []
for _ in range(100):
    x = np.random.randint(1, 100, size=(1, 2))
    x1 = x[0, 0]
    x2 = x[0, 1]

    if x1 > 50 and x2 > 50 or x1 < 50 and x2 < 50:
        target_value = np.array([(abs(x1 - x2))])
    else:
        target_value = np.array([(abs(x1 + x2 - 100))])
    target = np.array([target_value])
    training_data.append((x / SCALE, target / SCALE))

learning_rate = 0.1

limit_1 = np.sqrt(6 / 2)
w1 = np.random.uniform(-limit_1, limit_1, size=(2, 8))
bias1 = np.zeros((1, 8))

limit_2 = np.sqrt(6 / 8)
w2 = np.random.uniform(-limit_2, limit_2, size=(8, 16))
bias2 = np.zeros((1, 16))

limit_3 = np.sqrt(6 / 16)
w3 = np.random.uniform(-limit_3, limit_3, size=(16, 8))
bias3 = np.zeros((1, 8))

limit_output = np.sqrt(6 / 8)
w_output = np.random.uniform(-limit_output, limit_output, size=(8, 1))
output_bias = np.zeros((1, 1))

for epoch in range(10000):
    total_loss = 0.0

    for x, target in training_data:
        # Forward pass
        z1 = np.dot(x, w1) + bias1
        hidden_output_1 = leaky_relu(z1)

        z2 = np.dot(hidden_output_1, w2) + bias2
        hidden_output_2 = leaky_relu(z2)

        z3 = np.dot(hidden_output_2, w3) + bias3
        hidden_output_3 = leaky_relu(z3)

        old_output = np.dot(hidden_output_3, w_output) + output_bias

        error = target - old_output
        loss = error ** 2
        total_loss += loss

        # GRADIENTS
        d_loss = 2 * (old_output - target)
        
        gradient_w_output = d_loss * hidden_output_3.T
        gradient_bias_output = d_loss

        d_hidden_output_3 = d_loss.dot(w_output.T)
        d_z3 = d_hidden_output_3 * leaky_relu_derivative(z3)
        
        d_hidden_output_2 = d_z3.dot(w3.T)
        d_z2 = d_hidden_output_2 * leaky_relu_derivative(z2)

        d_hidden_output_1 = d_z2.dot(w2.T)
        d_z1 = d_hidden_output_1 * leaky_relu_derivative(z1)

        gradient_w3 = hidden_output_2.T.dot(d_z3)
        gradient_bias3 = d_z3

        gradient_w2 = hidden_output_1.T.dot(d_z2)
        gradient_bias2 = d_z2

        gradient_w1 = x.T.dot(d_z1)
        gradient_bias1 = d_z1

        # Update weights and biases
        w_output = w_output - learning_rate * gradient_w_output
        output_bias = output_bias - learning_rate * gradient_bias_output

        w3 = w3 - learning_rate * gradient_w3
        bias3 = bias3 - learning_rate * gradient_bias3

        w2 = w2 - learning_rate * gradient_w2
        bias2 = bias2 - learning_rate * gradient_bias2

        w1 = w1 - learning_rate * gradient_w1
        bias1 = bias1 - learning_rate * gradient_bias1


    average_loss = total_loss / len(training_data)
    if (epoch + 1) % 1000 == 0:
            print("Epoch:", epoch + 1, "Average Loss:", average_loss)


# NEW OUTPUT
for x, target in training_data:
    z1 = np.dot(x, w1) + bias1
    hidden_output_1 = leaky_relu(z1)
    
    z2 = np.dot(hidden_output_1, w2) + bias2
    hidden_output_2 = leaky_relu(z2)

    z3 = np.dot(hidden_output_2, w3) + bias3
    hidden_output_3 = leaky_relu(z3)

    output = np.dot(hidden_output_3, w_output) + output_bias

    loss = (target - output) ** 2
    average_loss = np.mean(loss)
    
    print("Input:", x * SCALE, "Target:", target * SCALE, "Prediction:", output * SCALE, "Loss:", average_loss)

# TESTING PHASE
x = np.array([[13, 7]]) / SCALE
target = np.array([[6]]) / SCALE

z1 = np.dot(x, w1) + bias1
hidden_output_1 = leaky_relu(z1)

z2 = np.dot(hidden_output_1, w2) + bias2
hidden_output_2 = leaky_relu(z2)

z3 = np.dot(hidden_output_2, w3) + bias3
hidden_output_3 = leaky_relu(z3)

output = np.dot(hidden_output_3, w_output) + output_bias

error = target - output
loss = error ** 2
average_loss = np.mean(loss)
print("Input:", x * SCALE, "Target:", target * SCALE, "Prediction:", output * SCALE, "Loss:", average_loss)

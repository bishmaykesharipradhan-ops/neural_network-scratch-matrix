import numpy as np

# TRAINING PHASE

SCALE = 100
learning_rate = 0.001

training_data = []
for _ in range(100):
    x = np.random.randint(1, 100, size=(1, 2))
    target = np.array([x[0, 0] + x[0, 1]])
    training_data.append((x/SCALE, target/SCALE))

w = np.random.uniform(-1, 1, size=(2, 1))
bias = np.zeros((1, 1))

for epoch in range(10000):
    total_loss = 0.0

    for x, target in training_data:
        output = np.dot(x, w) + bias
        error = target - output
        loss = error ** 2

        total_loss += np.mean(loss)

        w = w + learning_rate * error * x.T
        bias = bias + learning_rate * error

for x, target in training_data:
    output = np.dot(x, w) + bias
    loss = (target - output) ** 2
    print("Input:", x * SCALE, "Target:", target * SCALE, "Prediction:", output * SCALE, "Loss:", loss)

# TESTING PHASE
x = np.array([[343, 7]]) / SCALE
target = np.array([[350]]) / SCALE

output = np.dot(x, w) + bias
error = target - output
loss = error ** 2

print("Output:", output * SCALE, "Loss:", loss)

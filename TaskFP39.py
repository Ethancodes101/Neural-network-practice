import numpy as np

inputs = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
])

hidden_weights1 = np.array([
    [1,2],
    [3,4],
    [5,6]
])

def relu(x):
    return np.maximum(0,x)

hidden_bias1 = np.array([1,2,3])

hidden1 = np.dot(inputs, hidden_weights1.T) + hidden_bias1
print(hidden1)
relu_hidden1 = relu(hidden1)
hidden_weights2 = np.array([
    [1,2,3],
    [3,1,2]
])

hidden_bias2 = np.array([3,2])

hidden2 = np.dot(relu_hidden1, hidden_weights2.T) + hidden_bias2
print(hidden2)

relu_hidden2 = relu(hidden2)
output_weight = ([1,2])

output = np.dot(relu_hidden2, output_weight)
print(output)

print()

print(hidden1.shape)
print(hidden2.shape)
print(output.shape)
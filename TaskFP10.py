import numpy as np

inputs = np.array([1, 2])

weights1 = np.array([
    [1, 1],
    [2, 1],
    [0, 3]
])

biases1 = np.array([-5, 0, 1])

weights2 = np.array([
    [1, 2, 1]
])

biases2 = np.array([1])

def layer(inputs, weights, bias):
    return np.dot(weights, inputs) + bias

def relu(outputs):
    outputs[outputs < 0] = 0
    return outputs

def layer_relu(inputs, weights, bias):
    return relu(layer(inputs, weights, bias))

layer1_outputs = layer_relu(inputs, weights1, biases1)
final_output = layer(layer1_outputs, weights2, biases2)

print(layer1_outputs)
print(final_output)
import numpy as np
inputs = np.array([1, 2])

weights1 = np.array([1, 1])
weights2 = np.array([2, 1])
weights3 = np.array([0, 3])

bias1 = -5
bias2 = 0
bias3 = 1

output1 = np.dot(inputs, weights1) + bias1
output2 = np.dot(inputs, weights2) + bias2
output3 = np.dot(inputs, weights3) + bias3

layer1_outputs = np.array([output1, output2, output3])
layer1_outputs[layer1_outputs < 0] = 0
# layer1_outputs = [0, 4, 7]

weights4 = np.array([1, 2, 1])
bias4 = 1

final_output = np.dot(layer1_outputs, weights4) + bias4
print(final_output)
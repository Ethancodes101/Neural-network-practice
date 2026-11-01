import numpy as np
inputs = [
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
]

actuals = [
    10,
    15,
    20,
    25
]

w1 = np.random.randn() * 0.1
w2 = np.random.randn() * 0.1
w3 = np.random.randn() * 0.1
w4 = np.random.randn() * 0.1
w5 = np.random.randn() * 0.1
w6 = np.random.randn() * 0.1

b1 = 0
b2 = 0
b3 = 0

learning_rate = 0.001

def relu(x):
    return max(0,x)

for epoch in range(100):
    total_error = 0
    for i in range(len(inputs)):
        input1 = inputs[i][0]
        input2 = inputs[i][1]
        actual = actuals[i]

        h1 = relu((input1 * w1) + (input2 * w2) + b1)
        h2 = relu((input1 * w3) + (input2 * w4) + b2)
        print(epoch, h1, h2)

        output = (h1 * w5) + (h2 * w6) + b3

        error = output - actual

        gradient_w1 = error * input1 * w5
        gradient_w2 = error * input2 * w5
        gradient_w3 = error * input1 * w6
        gradient_w4 = error * input2 * w6

        gradient_w5 = error * h1
        gradient_w6 = error * h2

        gradient_b1 = error
        gradient_b2 = error
        gradient_b3 = error

        w1 -= learning_rate * gradient_w1
        w2 -= learning_rate * gradient_w2
        w3 -= learning_rate * gradient_w3
        w4 -= learning_rate * gradient_w4
        w5 -= learning_rate * gradient_w5
        w6 -= learning_rate * gradient_w6

        b1 -= learning_rate * gradient_b1
        b2 -= learning_rate * gradient_b2
        b3 -= learning_rate * gradient_b3

        total_error += abs(error)

    if epoch % 10 == 0:
        print(total_error)

print("\nWeights:")
print(w1,w2,w3,w4,w5,w6)
print("\nBiases:")
print(b1, b2, b3)
for i in range(len(inputs)):
    input1 = inputs[i][0]
    input2 = inputs[i][1]
    actual = actuals[i] 
    h1 = relu((input1*w1) + (input2*w2) + b1)
    h2 = relu((input1*w3) + (input2*w4) +b2)
    output = (h1*w5) + (h2*w6) + b3
    error = actual-output
    print(f"\nInputs: {input1}, {input2}")
    print(f"Hidden 1: {h1}")
    print(f"Hidden 2: {h2}")
    print(f"Actual: {actual}")
    print(f"Output: {output}")
    print(f"Error: {error}")
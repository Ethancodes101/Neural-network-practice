inputs = [
    [1, 1],
    [2, 1],
    [3, 1],
    [4, 1]
]

actuals = [
    7,
    12,
    17,
    22
]

w1 = 0
w2 = 0
bias = 0
learning_rate = 0.01

for epoch in range(200):
    total_error = 0
    for i in range(len(inputs)):
        input1 = inputs[i][0]
        input2 = inputs[i][1]
        actual = actuals[i]

        prediction = (input1*w1) + (input2*w2) + bias

        error = actual - prediction

        gradient_w1 = error * input1
        gradient_w2 = error * input2
        gradient_bias = error

        w1 += learning_rate * gradient_w1
        w2 += learning_rate * gradient_w2
        bias += learning_rate * gradient_bias
        total_error += abs(error)

    if epoch % 20 == 0:
        print(epoch, total_error)

print(f"\nW1: {w1}")
print(f"W2: {w2}")
print(f"Bias: {bias}")

for i in range(len(inputs)):
    input1 = inputs[i][0]
    input2 = inputs[i][1]
    actual = actuals[i] 
    prediction = (input1*w1) + (input2*w2) + bias
    error = actual-prediction
    print(f"\nInputs: {input1}, {input2}")
    print(f"Actual: {actual}")
    print(f"Prediction: {prediction}")
    print(f"Error: {error}")
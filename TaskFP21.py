import numpy as np

def predict(inputs, weight, bias=0):
    return (inputs*weight)+bias

def mse(actual, prediction):
    return np.mean((actual-prediction)**2)

inputs = np.array([1, 2, 3, 4])
actual = np.array([10, 20, 30, 40])

weight = 15
best_loss = mse(actual, predict(inputs, weight))

for i in range(101):
    current_loss = mse(actual, (predict(inputs, weight)))

    increase_weight = weight+0.1
    increase_loss = mse(actual, (predict(inputs, increase_weight)))
    
    lower_weight = weight - 0.1
    lower_loss = mse(actual, (predict(inputs, lower_weight)))

    if increase_loss < lower_loss and increase_loss < current_loss:
        weight = increase_weight
        best_loss = increase_loss
    elif lower_loss < increase_loss and lower_loss < current_loss:
        weight = lower_weight
        best_loss = lower_loss
    

print(best_loss)
print(weight)
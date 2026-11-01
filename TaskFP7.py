import numpy as np

def relu(outputs):
    outputs[outputs < 0] = 0
    return outputs
    #This changes the original

# def relu(outputs):
#     return np.maximum(0, outputs)
    #creates a new array

test = np.array([-5, 3, 0, 8])
print(relu(test))
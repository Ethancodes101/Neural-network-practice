import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)

print(model)
print(model[0].weight)
print(model[0].bias)
print()
print(model[2].weight)
print(model[2].bias)
print()

x = torch.tensor([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0]
])

output = model(x)
print(output)
print(output.shape)
import torch
import torch.nn as nn
import torch.optim as optim

x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0]
])

y = torch.tensor([
    [7.0],
    [12.0],
    [17.0],
    [22.0],
    [27.0]
])

model = nn.Sequential(
    nn.Linear(1,1)
)

optimizer = optim.SGD(
    model.parameters(),
    lr = 0.01
)

prediction = model(x)
loss = ((prediction-y)**2).mean()
counter = 1


while loss.item() > 0.001:

    optimizer.zero_grad()

    prediction = model(x)
    loss = ((prediction-y)**2).mean()

    loss.backward()

    optimizer.step()

    if counter % 100 == 0:
        print(counter, prediction, loss.item())

    counter += 1

final_predictions = model(x)
final_loss = ((final_predictions-y)**2).mean()

print(f"Epoch count: {counter}")
print(f"Final loss: {final_loss}")
print(f"Final weight: {model[0].weight.item()}")
print(f"Final bias: {model[0].bias.item()}")
print(f"Final predictions: {final_predictions}")
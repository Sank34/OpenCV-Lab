import torch as t
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2,8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8,1)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

x = t.tensor([[2.0,5.0]])
target = t.tensor([[8.0]])

model = SimpleModel()

optimizer = t.optim.SGD(
    model.parameters(),
    lr=0.001
)

criterion = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()

    pred = model(x)

    loss = criterion(pred, target)

    loss.backward()

    optimizer.step()

    print(epoch, loss.item())

# verify
print("Final pred:", model(x).item())
print("Target:", target.item())
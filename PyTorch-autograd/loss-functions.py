import torch as t
import torch.nn as nn

# MSE (Mean Squared Error)
prediction = t.tensor([3.0])
target = t.tensor([5.0])

criterion = nn.MSELoss()

loss = criterion(prediction, target)

print(loss) # 4.

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

model = SimpleModel()

criterion = nn.MSELoss()

x = t.tensor([[2.0, 5.0]])
target = t.tensor([[8.0]])

prediction = model(x)

loss = criterion(prediction, target)

print(loss)

loss.backward()

# for p in model.parameters():
#     print(p.grad)
for name, param in model.named_parameters():
    print(name)
    print("Weight:")
    print(param)
    print("Gradient:")
    print(param.grad)
import torch as t
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2,8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8,1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleModel()
x = t.tensor([[2.0,5.0]])
print(model)
print(model(x))
for p in model.parameters():
    print(p.shape)

print(sum(p.numel() for p in model.parameters()))

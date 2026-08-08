import torch.nn as nn
import torch as t

class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(
            2,
            1
        )

    def forward(self,x):
        x = self.fc(x)
        return x

# using our model
model = SimpleModel()

x = t.tensor([[2.0,5.0]])
pred = model(x)
print(pred)
print(model.fc.weight)
print(model.fc.bias)
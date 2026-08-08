import torch as t
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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

class WaySimplerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2,1)

    def forward(self,x):
        x = self.fc(x)
        return x

class RegressionDataset(Dataset):
    def __init__(self):
        self.inputs = t.tensor(
            [
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
                [5.0, 6.0],
                [6.0, 7.0],
            ],
            dtype=t.float32
        )

        self.targets = t.tensor(
            [
                [4.0],
                [7.0],
                [10.0],
                [13.0],
                [16.0],
                [19.0]
            ],
            dtype=t.float32
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

dataset = RegressionDataset()

print(len(dataset))
print(dataset[0])
print(dataset[2])

# alternatively we can use the TensorDataset class if we have the tensors already
# from torch.utils.data import TensorDataset -- import
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

model = WaySimplerModel()

criterion = nn.MSELoss()
optimizer = t.optim.Adam(
    model.parameters(),
    lr=0.01
)

# for inputs, targets in loader:
#     print("Inputs:")
#     print(inputs)
#
#     print("Targets:")
#     print(targets)

for epoch in range(2000): # by using the WaySimplerModel() model we can reduce the number of epochs, since it's just a liniar fct
    model.train()
    total_loss = 0.0

    for inputs, targets in loader:

        optimizer.zero_grad()

        preds = model(inputs)

        loss = criterion(preds, targets)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    if epoch % 20 == 0:
        print(
            f"Epoch {epoch:3d} |"
            f"Avg. loss: {avg_loss:.6f}"
        )

# eval model

model.eval()

test_input = t.tensor(
    [[7.0, 8.0]],
    dtype=t.float32
)

with t.no_grad():
    pred = model(test_input)

print("Prediction:", pred.item()) # pred: 21.999996185302734
print("Expected:", 22.0)
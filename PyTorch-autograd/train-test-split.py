import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

t.manual_seed(42)

class RegressionDataset(Dataset):
    def __init__(self, samples=100):
        self.inputs = t.rand(samples,2) * 10

        self.targets = (
            2 * self.inputs[:,0]
            + self.inputs[:,1]
        ).unsqueeze(1)

    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class SimplerModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(2,1)
    def forward(self,x):
        x = self.fc(x)
        return x

from torch.utils.data import random_split

dataset = RegressionDataset(samples=100)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = t.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

print(len(train_dataset))
print(len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

model = SimplerModel()

criterion = nn.MSELoss()

optimizer = t.optim.Adam(
    model.parameters(),
    lr = 0.01
)
for epoch in range(1000):
    model.train()

    train_loss = 0.0

    for inputs, targets in train_loader:
        optimizer.zero_grad()

        preds = model(inputs)

        loss = criterion(preds, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # validation
    model.eval()

    val_loss = 0.0
    with t.no_grad():
        for inputs, targets in val_loader:
            preds = model(inputs)
            loss = criterion(preds, targets)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    if epoch % 20 == 0:
        print(
            f"epoch: {epoch:3d} |"
            f"train: {avg_train_loss:.6f}|"
            f"valid: {avg_val_loss:.6f}"
        )

#metrics
all_preds = []
all_targets = []

model.eval()

with t.no_grad():
    for inputs, targets in val_loader:
        preds = model(inputs)

        all_preds.append(preds)
        all_targets.append(targets)

all_preds = t.cat(all_preds)
all_targets = t.cat(all_targets)
mse = t.mean(
    (all_preds - all_targets) ** 2
)
rmse = t.sqrt(mse)

mae = t.mean(t.abs(all_preds-all_targets))

print(mse.item())
print(rmse.item())
print(mae.item())
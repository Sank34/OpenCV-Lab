import torch as t
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

t.manual_seed(42)

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),
            # output = [ (W - F + 2P)/S ] + 1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32,32,kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*7*7,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# dataset

transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)
# test
# images, labels = next(iter(train_loader))
# print(images.shape)
# print(labels.shape)

# device
if t.cuda.is_available():
    device = t.device("cuda")
elif t.backends.mps.is_available():
    device = t.device("mps")
else:
    device = t.device("cpu")

# model
model = FashionCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 30
for epoch in range(epochs):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(imgs)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()

        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    print(
        f"Epoch {epoch+1}/{epochs}|"
        f"loss: {avg_loss:.4f}|"
        f"accuracy: {accuracy*100:.2f}%"
    )

# eval
model.eval()

test_loss = 0.0
correct = 0
total = 0
with t.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss = criterion(logits, labels)

        test_loss += loss.item() * imgs.size(0)

        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()

        total += labels.size(0)
avg_test_loss = test_loss / total
test_accuracy = correct / total

print(
    f"test loss: {avg_test_loss:.4f}| "
    f"accuracy: {test_accuracy*100:.2f}%"
)
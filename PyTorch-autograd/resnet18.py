import torch as t
import torch.nn as nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

for parameter in model.parameters():
    parameter.requires_grad = False

input_features = model.fc.in_features

model.fc = nn.Linear(
    input_features,
    3
)
print(model.fc)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)

dummy_input = t.randn(
    4,
    3,
    224,
    224
)

output = model(dummy_input)

print("Input shape", dummy_input.shape)
print("Output shape:", output.shape)
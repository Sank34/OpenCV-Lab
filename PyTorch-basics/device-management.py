import torch as t

x = t.tensor([1,2,3])
print(x.device)
# x = x.to("mps")
# print(x.device)

if t.cuda.is_available():
    device = "cuda"
elif t.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

x = x.to(device)
print(x.device)

x_cpu = x.cpu()

print(x_cpu.device)
print(x_cpu.numpy())
# alternativ
y = t.tensor([1,2,3,4], device=device)

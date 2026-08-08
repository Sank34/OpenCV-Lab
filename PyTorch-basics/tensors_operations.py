import torch as t

x = t.tensor([
    [1,2],
    [3,4]
], dtype=t.float32)

y = t.tensor([
    [5,6],
    [7,8]
], dtype=t.float32)

print(x+y)
print(x - y)
print(x * y)
print(x / y)
print(x ** 2)
print(x @ y)
print(t.matmul(x,y))
print(x.sum())
print(x.mean())
print(x.max())
print(x.min())
print("------")
z = t.arange(10)
print(z.reshape(2,5))
print(z.reshape(-1))
print(z.reshape(2,5).transpose(0,1))
print(z.reshape(1,2,5).permute(2,0,1))
print(z.unsqueeze(0).shape)
print(z.squeeze().shape)
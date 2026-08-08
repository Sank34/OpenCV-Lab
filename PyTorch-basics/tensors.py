import torch as t

x1 = t.tensor(5) #0D - scalar
print(x1.shape)

x2 = t.tensor([1,2,3]) # 1D
print(x2.shape)

x3 = t.tensor([
    [1,2],
    [3,4]
]) # 2D
print(x3.shape)

x4 = t.tensor([
    [
        [1,2],
        [3,4]
    ],
    [
        [5,6],
        [7,8]
    ]
])
print(x4.shape) # 3D

print(t.zeros((3,4)))
print(t.ones((3,4)))
print(t.rand((3,4)))
print(t.randn((3,4)))
print(t.arange(10))
print(t.eye(3))
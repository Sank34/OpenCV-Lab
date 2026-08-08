import torch as t

# Atomatic Differentiation (Autograd) refers to the concept of calculating the derivative (gradient)
# of a function in order to figure out how to modify the params to reduce the error rate

x = t.tensor(
    2.0,
    requires_grad=True
)
y = x ** 2
# x -> square -> y
# for more complex functions:
# y = f(g(x)) - chain rule
# dy/dx = dy/dg * dg/dx
# now let's use backpropagation
y.backward()
print(x.grad) # 4.

# note: after running the backward() fc the computational graph is freed from memory

# z = y + 3
# w = z * 5

# Exercise 1

x1 = t.tensor(2.0, requires_grad=True)
y1 = x1 ** 2
y1.backward()
print(x1.grad) # 4.

# Exercise 2
x2 = t.tensor(3.0, requires_grad=True)
y2 = 5 * x2 + 1
y2.backward()
print(x2.grad) # 5.

# Exercise 3
x3 = t.tensor(2.0, requires_grad=True)
y3 = x3 ** 3 + 2 * x3
y3.backward()
print(x3.grad) # 14.

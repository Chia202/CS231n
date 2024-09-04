import autograd.numpy as gnp
from autograd import grad

# Define a function like normal, using Python and (autograd's) NumPy
def tanh(x):
    y = gnp.exp(-x)
    return (1.0 - y) / (1.0 + y)


# Create a *function* that computes the gradient of tanh
grad_tanh = grad(tanh)

# Evaluate the gradient at x = 1.0
print(grad_tanh(1.0))

# Compare to numeric gradient computed using finite differences
print((tanh(1.0001) - tanh(0.9999)) / 0.0002)

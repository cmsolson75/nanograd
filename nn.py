import numpy as np

from engine import Tensor

class Module:
    def zero_grad(self):
        for param in self.parameters():
            if param.requires_grad:
                param.grad = np.zeros_like(param.data)

    def parameters(self):
        for name, value in vars(self).items():
            if isinstance(value, Tensor):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()

class Linear(Module):
    def __init__(self, in_dims, out_dims):
        self.w = Tensor.kaiming_uniform(in_dims, out_dims, requires_grad=True)
        self.b = Tensor.zeros((1, out_dims), requires_grad=True)
    
    def __call__(self, x):
        return x@self.w + self.b

    def __repr__(self):
        return f"Linear({self.w.shape})"


# Need to orchestrate them

class MLP(Module):
    def __init__(self):
        self.layer1 = Linear(3, 4)
        self.layer2 = Linear(4, 4)
        self.layer3 = Linear(4, 4)
        self.layer4 = Linear(4, 1)
    
    def __call__(self, x):
        # x = x.flatten()# should add something like this
        x = self.layer1(x).tanh()
        x = self.layer2(x).tanh()
        x = self.layer3(x).tanh()
        x = self.layer4(x)
        return x

class Optim:
    pass

class SGD(Optim):
    pass
    # Probably put optimizers in there own file

class Adam(Optim):
    pass

class AdamW(Optim):
    pass

class Trainer:
    pass


# This could be how to do it
# have a simple 


# I need it to do a sequential like thing

# self.layers = nn.Sequential
# I need it to keep track of its layers
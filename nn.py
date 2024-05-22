import numpy as np
from tensor import Tensor
import math

class Module:
    def parameters(self):
        for name, value in vars(self).items():
            if isinstance(value, Tensor):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()
    
    def __call__(self, x):
        self.forward(x)
    
    def forward(self, x):
        raise NotImplementedError

class Linear(Module):
    def __init__(self, in_dims, out_dims):
        self.weight = Tensor.kaiming_uniform(in_dims, out_dims, gain=math.sqrt(5), requires_grad=True)
        self.bias = Tensor.zeros((1, out_dims), requires_grad=True)
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        return x @ self.weight + self.bias

    def __repr__(self):
        return f"Linear({self.w.shape})"
    

class Sequential(Module):
    pass


class Conv1d(Module):
    pass

class Conv2d(Module):
    pass

class MaxPool1d(Module):
    pass

class MaxPool2d(Module):
    pass

class AvgPool1d(Module):
    pass

class AvgPool2d(Module):
    pass

#------------------------------

class Optimizer:
    def __init__(self, parameters, lr, weight_decay=0):
        self.parameters = list(parameters)
        self.learning_rate = lr
        self.weight_decay = weight_decay

    def zero_grad(self, lazy=True):
        for param in self.parameters:
            if param.requires_grad:
                if lazy:
                    param.grad = None
                else:
                    param.grad = np.zeros_like(param.data)

    def step(self):
        raise NotImplementedError("This method should be overridden by subclasses")

class SGD(Optimizer):
    """<https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>"""
    def __init__(self, parameters, lr, momentum=0, weight_decay=0, dampening=0, nesterov=False):
        super().__init__(parameters, lr, weight_decay)
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        self.velocity = [None for p in self.parameters if p.requires_grad]

    def step(self):
        for idx, param in enumerate(self.parameters):
            if param.grad is not None:
                grad = param.grad
                if self.weight_decay != 0:
                    grad += self.weight_decay * param.data

                if self.momentum != 0:
                    if self.velocity[idx] is None:
                        self.velocity[idx] = grad
                    else:
                        self.velocity[idx] = self.momentum * self.velocity[idx] + (1 - self.dampening) * grad

                    if self.nesterov:
                        grad += self.momentum * self.velocity[idx]
                    else:
                        grad = self.velocity[idx]
                
                param.data -= self.learning_rate * grad




class RMSProp(Optimizer):
    pass

class Adam(Optimizer):
    pass

class AdamW(Adam): # Probably extends the adam class
    pass

class ADAGrad(Optimizer):
    pass

# -----------------------------------------

class Dataset:
    pass

class MNIST(Dataset):
    pass

class Cifar10(Dataset):
    pass


# ------------------------------------------

class DataLoader:
    pass

# ------------------------------------------

class Loss:
    def __call__(self, y_hat, y):
        raise NotImplementedError

class NLLLoss(Loss):
    def __call__(self, y_hat, y):
        N = y_hat.data.shape[0]  # Batch size
        log_probs = y_hat  # Assuming y_hat is already log_softmax output

        # Gather the log probabilities corresponding to the true labels
        nll = (-log_probs[np.arange(N), y.data]).mean()  # This is masking for output
        return nll
    

class MSELoss(Loss):
    def __call__(self, y_hat, y):
        loss = (y_hat - y).square().mean()
        return loss

class BCELoss(Loss):
    def __call__(self, y_hat, y):
        # Ensure numerical stability with epsilon
        epsilon = 1e-12
        y_hat = y_hat.clip(epsilon, 1 - epsilon)
        
        # Calculate the BCE loss
        loss = -(y * y_hat.log() + (1 - y) * (1 - y_hat).log()).mean()
        return loss


class CrossEntropyLoss(Loss):
    # Sparse Catagorical Cross Entropy Loss
    def __call__(self, y_hat, y):
        log_prob = y_hat.log_softmax(axis=1) # need to check axis
        loss = NLLLoss()(log_prob, y)
        return loss


class L1Loss(Loss):
    def __call__(self, y_hat, y):
        loss = (y_hat - y).abs().mean()


class HuberLoss:
    pass



#---------------

class BatchNorm1d(Module):
    pass

class BatchNorm2d(Module):
    pass

class Dropout(Module):
    pass


#-----------------

class ReLU(Module):
    pass

class LeakyReLU(Module):
    pass

class Tanh(Module):
    pass

class Sigmoid(Module):
    pass
import numpy as np
from tensor import Tensor
import math


class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def parameters(self):
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def add_parameter(self, name, param):
        self._parameters[name] = param

    def add_module(self, name, module):
        self._modules[name] = module

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class Linear(Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.weight = Tensor.kaiming_uniform(
            in_dims, out_dims, gain=math.sqrt(5), requires_grad=True
        )
        self.bias = Tensor.zeros((1, out_dims), requires_grad=True)
        self.add_parameter('weight', self.weight)
        self.add_parameter('bias', self.bias)

    def forward(self, x):
        return x @ self.weight + self.bias

    def __repr__(self):
        return f"Linear(in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]})"


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self._modules = {str(i): module for i, module in enumerate(modules)}

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    def __repr__(self):
        module_reprs = [f'({i}): {module}' for i, module in self._modules.items()]
        return 'Sequential(\n  ' + '\n  '.join(module_reprs) + '\n)'


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


# ------------------------------


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
                        self.velocity[idx] = (
                            self.momentum * self.velocity[idx] + (1 - self.dampening) * grad
                        )

                    if self.nesterov:
                        grad += self.momentum * self.velocity[idx]
                    else:
                        grad = self.velocity[idx]

                param.data -= self.learning_rate * grad


class RMSProp(Optimizer):
    pass


class Adam(Optimizer):
    pass


class AdamW(Adam):  # Probably extends the adam class
    pass


class ADAGrad(Optimizer):
    pass


# -----------------------------------------


class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class MNIST(Dataset):
    pass


class Cifar10(Dataset):
    pass


# ------------------------------------------

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration
        
        start_index = self.current_index
        end_index = min(self.current_index + self.batch_size, len(self.indices))
        batch_indices = self.indices[start_index:end_index]
        batch = [self.dataset[i] for i in batch_indices]
        
        self.current_index = end_index
        return batch

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


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
    def __init__(self, reduction: str = 'mean'):
        """
        Initializes the BCELoss with the specified reduction method.
        
        Args:
            reduction (str): Specifies the reduction to apply to the output: 'mean' | 'sum' | 'none'.
        """
        self.reduction = reduction
        
    def __call__(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """
        Computes the Binary Cross-Entropy loss.
        
        Args:
            y_hat (Tensor): Predicted probabilities.
            y (Tensor): Ground truth labels.
        
        Returns:
            Tensor: The computed loss.
        """
        epsilon = 1e-6  # Small value to prevent log(0)
        y_hat = y_hat.clip(epsilon, 1 - epsilon)  # Ensure y_hat is in a stable range
        loss = -(y * y_hat.log() + (1 - y) * (1 - y_hat).log())
        
        if self.reduction == 'mean':
            reduced_loss = loss.mean()
        elif self.reduction == 'sum':
            reduced_loss = loss.sum()
        else:
            reduced_loss = loss
        
        return reduced_loss


class CrossEntropyLoss(Loss):
    def __call__(self, y_hat, y):
        log_prob = y_hat.log_softmax(axis=1)
        loss = NLLLoss()(log_prob, y)
        return loss


class L1Loss(Loss):
    def __call__(self, y_hat, y):
        loss = (y_hat - y).abs().mean()


class HuberLoss:
    pass


# ---------------


class BatchNorm1d(Module):
    pass


class BatchNorm2d(Module):
    pass


class Dropout(Module):
    pass


# -----------------


class ReLU(Module):
    def __call__(self, x):
        return x.relu()


class LeakyReLU(Module):
    pass


class Tanh(Module):
    def __call__(self, x):
        return x.tanh()


class Sigmoid(Module):
    def __call__(self, x):
        return x.sigmoid()

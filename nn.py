import numpy as np
from typing import Iterable, Generator, Optional, Tuple, List
from tensor import Tensor
import math


class Module:
    """
    Base class for all neural network modules.
    """

    def parameters(self) -> Generator[Tensor, None, None]:
        """
        Returns an iterator over module parameters.

        Yields
        ------
        Tensor
            A parameter tensor of the module.
        """
        for name, value in vars(self).items():
            if isinstance(value, Tensor):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()

    def __call__(self, x: Tensor) -> Tensor:
        """
        Makes the module callable.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the computation performed at every call.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        raise NotImplementedError


class Linear(Module):
    """
    Linear layer (fully connected layer).
    """

    def __init__(self, in_dims: int, out_dims: int) -> None:
        """
        Initializes the linear layer with the given dimensions.

        Args:
            in_dims (int): Number of input dimensions.
            out_dims (int): Number of output dimensions.
        """
        self.weight = Tensor.kaiming_uniform(
            in_dims, out_dims, gain=math.sqrt(5), requires_grad=True
        )
        self.bias = Tensor.zeros((1, out_dims), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass of the linear layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return x @ self.weight + self.bias

    def __repr__(self) -> str:
        return f"Linear({self.weight.shape})"




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


# ------------------------------


class Optimizer:
    """
    Base class for all optimizers.
    """

    def __init__(self, parameters: Iterable[Tensor], lr: float, weight_decay: float = 0) -> None:
        """
        Initializes the optimizer.

        Args:
            parameters (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate.
            weight_decay (float): Weight decay (L2 penalty). Default is 0.
        """
        self.parameters = list(parameters)
        self.learning_rate = lr
        self.weight_decay = weight_decay

    def zero_grad(self, lazy: bool = True) -> None:
        """
        Sets gradients of all parameters to zero.

        Args:
            lazy (bool): If True, sets gradients to None. Otherwise, sets gradients to zero arrays. Default is True.
        """
        for param in self.parameters:
            if param.requires_grad:
                if lazy:
                    param.grad = None
                else:
                    param.grad = np.zeros_like(param.data)

    def step(self) -> None:
        """
        Performs a single optimization step.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses")


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with optional momentum, weight decay, dampening, and Nesterov momentum.
    <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>
    """

    def __init__(
        self, parameters: Iterable[Tensor], lr: float, momentum: float = 0, weight_decay: float = 0, dampening: float = 0, nesterov: bool = False
    ) -> None:
        """
        Initializes the SGD optimizer.

        Args:
            parameters (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate.
            momentum (float): Momentum factor. Default is 0.
            weight_decay (float): Weight decay (L2 penalty). Default is 0.
            dampening (float): Dampening for momentum. Default is 0.
            nesterov (bool): Enables Nesterov momentum. Default is False.
        """
        super().__init__(parameters, lr, weight_decay)
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        self.velocity = [None for p in self.parameters if p.requires_grad]

    def step(self) -> None:
        """
        Performs a single optimization step.
        """
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
                            self.momentum * self.velocity[idx]
                            + (1 - self.dampening) * grad
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
    """
    Base class for all loss functions.
    """

    # Should make a general reduction function due to all losses using it.
    # Should probably make forward API like Module Has

    def __call__(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """
        Computes the loss between predictions and targets.

        Args:
            y_hat (Tensor): Predicted values.
            y (Tensor): True values.

        Returns:
            Tensor: Computed loss.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError


class NLLLoss(Loss):
    """
    Negative Log Likelihood loss.
    """

    def __call__(self, y_hat: Tensor, y: Tensor) -> float:
        """
        Computes the Negative Log Likelihood loss.

        Args:
            y_hat (Tensor): Log probabilities.
            y (Tensor): True labels.

        Returns:
            float: Computed loss.
        """
        N = y_hat.data.shape[0]
        log_probs = y_hat
        nll = (-log_probs[np.arange(N), y.data]).mean()
        return nll


class MSELoss(Loss):
    """
    Mean Squared Error loss.
    """
    def __init__(self, reduction: str = 'mean') -> None:
        self.reduction = reduction

    def __call__(self, y_hat: Tensor, y: Tensor) -> float:
        """
        Computes the Mean Squared Error loss.

        Args:
            y_hat (Tensor): Predicted values.
            y (Tensor): True values.

        Returns:
            float: Computed loss.
        """
        loss = (y_hat - y).square()

        if self.reduction == 'mean':
            reduced_loss = loss.mean()
        elif self.reduction == 'sum':
            reduced_loss = loss.sum()
        else:
            reduced_loss = loss

        return reduced_loss


class BCELoss(Loss):
    """
    Binary Cross-Entropy loss.
    """

    def __init__(self, reduction: str = 'mean') -> None:
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
        epsilon = 1e-6
        y_hat = y_hat.clip(epsilon, 1 - epsilon)
        loss = -(y * y_hat.log() + (1 - y) * (1 - y_hat).log())

        if self.reduction == 'mean':
            reduced_loss = loss.mean()
        elif self.reduction == 'sum':
            reduced_loss = loss.sum()
        else:
            reduced_loss = loss

        return reduced_loss


class CrossEntropyLoss(Loss):
    """
    Cross-Entropy loss.
    """

    def __call__(self, y_hat: Tensor, y: Tensor) -> float:
        """
        Computes the Cross-Entropy loss.

        Args:
            y_hat (Tensor): Logits.
            y (Tensor): True labels.

        Returns:
            float: Computed loss.
        """
        log_prob = y_hat.log_softmax(axis=1)
        loss = NLLLoss()(log_prob, y)
        return loss


class L1Loss(Loss):
    """
    L1 loss (Mean Absolute Error).
    """

    def __call__(self, y_hat: Tensor, y: Tensor) -> float:
        """
        Computes the L1 loss.

        Args:
            y_hat (Tensor): Predicted values.
            y (Tensor): True values.

        Returns:
            float: Computed loss.
        """
        loss = (y_hat - y).abs().mean()
        return loss

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
